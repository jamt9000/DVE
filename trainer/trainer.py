import numpy as np
import torch
import time
import datetime
from torchvision.utils import make_grid
from pkg_resources import parse_version
from base import BaseTrainer
from torch.nn.modules.batchnorm import _BatchNorm
from model.metric import runningIOU


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(
            self,
            model,
            loss,
            metrics,
            optimizer,
            resume,
            config,
            data_loader,
            valid_data_loader=None,
            lr_scheduler=None,
            visualizations=None,
            mini_train=False,
            check_bn_working=False,
            **kwargs,
    ):
        super().__init__(model, loss, metrics, optimizer, resume, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = max(1, int(len(self.data_loader) / 5.))
        self.visualizations = visualizations if visualizations is not None else []
        self.mini_train = mini_train
        self.check_bn_working = check_bn_working
        self.loss_args = config.get('loss_args', {})

        assert self.lr_scheduler.optimizer is self.optimizer
        assert self.start_epoch >= 1

        if self.start_epoch != 1:
            # Our epoch 1 is step -1 (but we can't explicitly call
            # step(-1) because that would update the lr based on a negative epoch)
            # NB stateful schedulers eg based on loss won't be restored properly
            self.lr_scheduler.step(self.start_epoch - 2)

        # only perform last epoch check for older PyTorch, current versions
        # immediately set the `last_epoch` attribute to 0.
        if parse_version(torch.__version__) <= parse_version("1.0.0"):
            assert self.lr_scheduler.last_epoch == self.start_epoch - 2

        print('Loss args', self.loss_args)

        class LossWrapper(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def __call__(self, *a, **kw):
                return self.fn(*a, **kw)

        if isinstance(self.model, torch.nn.DataParallel):
            self.loss_wrapper = torch.nn.DataParallel(
                LossWrapper(self.loss),
                device_ids=self.model.device_ids
            )

        if self.config.get('cache_descriptors', False):
            cache_tic = time.time()
            self.model.eval()
            datasets = {
                "train": self.data_loader.dataset,
                "val": self.valid_data_loader.dataset,
            }
            self.cache = {}
            # First determine the spatial/depth dimensions of the tensor cache to enable
            # preallocation of the tensor caches
            init_batcher = torch.utils.data.DataLoader(datasets["train"], batch_size=1)
            with torch.no_grad():
                batch = next(iter(init_batcher))
                dd, mm = batch["data"], batch["meta"]
                feat_shape = self.model[0].forward(dd.to(self.device))[0].shape

            for key, dataset in datasets.items():
                batcher = torch.utils.data.DataLoader(dataset, batch_size=100)
                self.cache[key] = torch.zeros(len(dataset), *feat_shape[1:])
                for ii, batch in enumerate(batcher):
                    with torch.no_grad():
                        dd, mm = batch["data"], batch["meta"]
                        fw = self.model[0].forward(dd.to(self.device))[0].to('cpu')
                        self.cache[key][mm["index"]] = fw
                    self.logger.info(f"caching {key} descriptors {ii}/{len(batcher)}")

            duration = time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - cache_tic))
            self.logger.info(f"Descriptor caching took {duration}")
            self.model.train()

        # Handle segmentation metrics separately, since the accumulation cannot be
        # done directly via an AverageMeter
        self.log_miou = self.config["trainer"].get("log_miou", False)

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(
                output,
                target,
                self.data_loader.dataset,
                self.config
            )
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    # def printer(self, msg):
    #     print("{:.3f} >>> {}".format(time.time() - self.tic, msg))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        train_tic = time.time()
        avg_loss = AverageMeter()
        total_metrics = [AverageMeter() for a in range(len(self.metrics))]
        seen_tic = time.time()
        seen = 0
        profile = self.config["profile"]
        totaL_batches = len(self.data_loader)

        if profile:
            batch_tic = time.time()
        for batch_idx, batch in enumerate(self.data_loader):
            data, meta = batch["data"], batch["meta"]
            data = data.to(self.device)
            seen += data.shape[0] // 2

            if profile:
                timings = {}
                timings["data transfer"] = time.time() - batch_tic
                tic = time.time()

            self.optimizer.zero_grad()
            if self.config.get('cache_descriptors', False):
                assert isinstance(self.model, torch.nn.Sequential)
                descs = self.cache["train"][meta['index']].to(self.device).float()
                output = self.model[1:]([descs])
            else:
                output = self.model(data)

            if profile:
                timings["fwd"] = time.time() - tic
                tic = time.time()

            if isinstance(self.model, torch.nn.DataParallel):
                loss = self.loss_wrapper(
                    output,
                    meta,
                    **self.loss_args
                )
                loss = loss.mean()
            else:
                loss = self.loss(
                    output,
                    meta,
                    **self.loss_args
                )
            if profile:
                timings["loss-fwd"] = time.time() - tic
                tic = time.time()

            loss.backward()

            if profile:
                timings["loss-back"] = time.time() - tic
                tic = time.time()

            self.optimizer.step()
            if profile:
                timings["optim-step"] = time.time() - tic
                tic = time.time()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            avg_loss.update(loss.item(), data.size(0))

            for i, m in enumerate(self._eval_metrics(output, meta)):
                total_metrics[i].update(m, data.size(0))

            if profile:
                timings["metrics"] = time.time() - tic

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                toc = time.time() - seen_tic
                rate = seen / max(toc, 1E-5)
                tic = time.time()
                msg = "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} "
                msg += "Hz: {:.2f}, ETA: {}"
                batches_left = totaL_batches - batch_idx
                remaining = batches_left * self.data_loader.batch_size / rate
                eta_str = str(datetime.timedelta(seconds=remaining))
                self.logger.info(
                    msg.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        len(self.data_loader.dataset),
                        100.0 * batch_idx / len(self.data_loader),
                        loss.item(),
                        rate,
                        eta_str
                    )
                )

                # Use key check for backward compat
                if "im_data" in batch:
                    im_data = batch["im_data"]
                else:
                    im_data = data
                if len(im_data) and self.visualizations:
                    im = make_grid(im_data.cpu(), nrow=8, normalize=True)
                    self.writer.add_image('input', im)
                    for v in self.visualizations:
                        v(self.writer, im_data.cpu(), output, meta)
                seen_tic = time.time()
                seen = 0
                if profile:
                    timings["vis"] = time.time() - tic

            # Do some aggressive reference clearning to ensure that we don't
            # hang onto memory while fetching the next minibatch (for safety, disabling
            # this for now)

            # del data
            # del loss
            # del output

            if profile:
                timings["minibatch"] = time.time() - batch_tic
                batch_tic = time.time()

                print("==============")
                for key in timings:
                    ratio = 100 * timings[key] / timings["minibatch"]
                    msg = "{:.3f} ({:.2f}%) >>> {}"
                    print(msg.format(timings[key], ratio, key))
                print("==============")

            if self.mini_train and batch_idx > 3:
                self.logger.info("Mini training: exiting epoch early...")
                break

        log = {'loss': avg_loss.avg, 'metrics': [a.avg for a in total_metrics]}
        self.writer.set_step(epoch, 'train_epoch')
        self.writer.add_scalar('loss', log['loss'])

        for i, metric in enumerate(self.metrics):
            self.writer.add_scalar(f'{metric.__name__}', log['metrics'][i])

        duration = time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - train_tic))
        print(f"training epoch took {duration}")

        val_tic = time.time()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}
        duration = time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - val_tic))
        print(f"validation epoch took {duration}")

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch - 1)
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        cached_state = torch.get_rng_state()
        self.logger.info(f"Running validation for epoch {epoch}")
        self.model.eval()
        avg_val_loss = AverageMeter()
        total_val_metrics = [AverageMeter() for a in range(len(self.metrics))]

        if self.log_miou:
            nclass = self.config["segmentation_head"]["args"]["num_classes"]
            running_metrics_val = runningIOU(nclass)

        with torch.no_grad():
            torch.manual_seed(0)
            for batch_idx, batch in enumerate(self.valid_data_loader):
                data, meta = batch["data"], batch["meta"]
                data = data.to(self.device)

                if self.config.get('cache_descriptors', False):
                    assert isinstance(self.model, torch.nn.Sequential)
                    descs = self.cache["val"][meta['index']].to(self.device).float()
                    output = self.model[1:]([descs])
                else:
                    output = self.model(data)
                # output = self.model(data)

                if isinstance(self.model, torch.nn.DataParallel):
                    loss = self.loss_wrapper(output, meta, **self.loss_args)
                    loss = loss.mean()
                else:
                    loss = self.loss(output, meta, **self.loss_args)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader)
                                     + batch_idx,
                                     'valid')
                self.writer.add_scalar('loss', loss.item())
                avg_val_loss.update(loss.item(), data.size(0))
                for i, m in enumerate(self._eval_metrics(output, meta)):
                    total_val_metrics[i].update(m, data.size(0))

                # Use key check for backward compat
                if "im_data" in batch:
                    im_data = batch["im_data"]
                else:
                    im_data = data
                if len(im_data) and self.visualizations:
                    im_grid = make_grid(im_data.cpu(), nrow=8, normalize=True)
                    self.writer.add_image('input', im_grid)
                    for v in self.visualizations:
                        v(self.writer, im_data.cpu(), output, meta)

                if self.mini_train and batch_idx > 3:
                    self.logger.info("Mini training: exiting validation epoch early...")
                    break

        val_log = {
            'val_loss': avg_val_loss.avg,
            'val_metrics': [a.avg for a in total_val_metrics]
        }

        self.writer.set_step(epoch, 'val_epoch')
        self.writer.add_scalar('val_loss', val_log['val_loss'])

        if self.check_bn_working:
            # Run without using saved batchnorm statistics, to check bn is working
            for md in self.model.modules():
                if isinstance(md, _BatchNorm):
                    md.track_running_stats = False

            avg_val_loss_trainbn = AverageMeter()
            with torch.no_grad():
                torch.manual_seed(0)
                for batch_idx, batch in enumerate(self.valid_data_loader):
                    data, meta = batch["data"], batch["meta"]
                    data = data.to(self.device)

                    output = self.model(data)

                    if isinstance(self.model, torch.nn.DataParallel):
                        loss = self.loss_wrapper(output, meta, **self.loss_args)
                        loss = loss.mean()
                    else:
                        loss = self.loss(output, meta, **self.loss_args)

                    avg_val_loss_trainbn.update(loss.item(), data.size(0))

                    if self.log_miou:
                        running_metrics_val.update(output, meta)

            for md in self.model.modules():
                if isinstance(md, _BatchNorm):
                    md.track_running_stats = True
            self.writer.add_scalar('val_loss_trainbn', val_log['val_loss_trainbn'])
            val_log['val_loss_trainbn'] = avg_val_loss_trainbn.avg

        if self.log_miou:
            summary, cliu = running_metrics_val.get_scores()
            for key, val in summary.items():
                self.writer.add_scalar('seg/{}'.format(key), val)
                print("{}: {}".format(key, val))
            for cls, val in cliu.items():
                clsname = self.valid_data_loader.dataset.classnames[cls]
                self.writer.add_scalar('seg/cliu-{}'.format(clsname), val)

        for i, metric in enumerate(self.metrics):
            self.writer.add_scalar(f'{metric.__name__}', val_log['val_metrics'][i])

        torch.set_rng_state(cached_state)

        return val_log
