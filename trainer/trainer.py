import numpy as np
import torch
import time
import datetime
from torchvision.utils import make_grid
from base import BaseTrainer
from torch.nn.modules.batchnorm import _BatchNorm


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

    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None,
                 train_logger=None, visualizations=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume,
                                      config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = 2 * int(np.sqrt(data_loader.batch_size))
        self.visualizations = visualizations if visualizations is not None else []
        self.loss_args = config.get('loss_args', {})

        print('Loss args', self.loss_args)

        class LossWrapper(torch.nn.Module):
            def __init__(self, fn):
                super(LossWrapper, self).__init__()
                self.fn = fn

            def __call__(self, *a, **kw):
                return self.fn(*a, **kw)

        self.loss_wrapper = torch.nn.DataParallel(LossWrapper(self.loss), device_ids=self.model.device_ids)

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target, self.data_loader.dataset, self.config)
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    def printer(self, msg):
        print("{:.3f} >>> {}".format(time.time() - self.tic, msg))

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

        avg_loss = AverageMeter()
        total_metrics = [AverageMeter() for a in range(len(self.metrics))]
        seen_tic = time.time()
        seen = 0
        profile = self.config["profile"]
        totaL_batches = len(self.data_loader)
        for batch_idx, (data, meta) in enumerate(self.data_loader):
            if profile:
                timings = {}
                batch_tic = time.time()

            data = data.to(self.device)
            seen += data.shape[0] // 2

            if profile:
                timings["data transfer"] = time.time() - batch_tic
                tic = time.time()

            self.optimizer.zero_grad()
            output = self.model(data)

            if profile:
                timings["fwd"] = time.time() - tic
                tic = time.time()

            if isinstance(self.model, torch.nn.DataParallel):
                loss = self.loss_wrapper(output, meta, fold_corr=self.config["fold_corr"], **self.loss_args)
                loss = loss.mean()
            else:
                loss = self.loss(output, meta,
                                 fold_corr=self.config["fold_corr"], **self.loss_args)
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
                self.logger.info(msg.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    len(self.data_loader.dataset),
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item(),
                    rate,
                    eta_str))
                im = make_grid(data.cpu(), nrow=8, normalize=True)
                self.writer.add_image('input', im)
                for v in self.visualizations:
                    v(self.writer, data.cpu(), output, meta)
                seen_tic = time.time()
                seen = 0
                if profile:
                    timings["vis"] = time.time() - tic

            """Do some aggressive reference clearning to ensure that we don't
            hang onto memory while fetching the next minibatch."""
            #  For safety, disabling this for now
            # del data
            # del loss
            # del output

            if profile:
                timings["minibatch"] = time.time() - batch_tic

                print("==============")
                for key in timings:
                    ratio = 100 * timings[key] / timings["minibatch"]
                    msg = "{:.3f} ({:.2f}%) >>> {}"
                    print(msg.format(timings[key], ratio, key))
                print("==============")

        log = {
            'loss': avg_loss.avg,
            'metrics': [a.avg for a in total_metrics]
        }

        self.writer.set_step(epoch, 'train_epoch')
        self.writer.add_scalar('loss', log['loss'])

        for i, metric in enumerate(self.metrics):
            self.writer.add_scalar(f'{metric.__name__}', log['metrics'][i])

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        cached_state = torch.get_rng_state()
        self.model.eval()
        avg_val_loss = AverageMeter()
        total_val_metrics = [AverageMeter() for a in range(len(self.metrics))]
        with torch.no_grad():
            torch.manual_seed(0)
            for batch_idx, (data, meta) in enumerate(self.valid_data_loader):
                data = data.to(self.device)

                output = self.model(data)

                if isinstance(self.model, torch.nn.DataParallel):
                    loss = self.loss_wrapper(output, meta, **self.loss_args)
                    loss = loss.mean()
                else:
                    loss = self.loss(output, meta, **self.loss_args)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                avg_val_loss.update(loss.item(), data.size(0))
                for i, m in enumerate(self._eval_metrics(output, meta)):
                    total_val_metrics[i].update(m, data.size(0))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                for v in self.visualizations:
                    v(self.writer, data.cpu(), output, meta)

        # Run without using saved batchnorm statistics, to check bn is working
        for md in self.model.modules():
            if isinstance(md, _BatchNorm):
                md.track_running_stats = False

        avg_val_loss_trainbn = AverageMeter()
        with torch.no_grad():
            torch.manual_seed(0)
            for batch_idx, (data, meta) in enumerate(self.valid_data_loader):
                data = data.to(self.device)

                output = self.model(data)

                if isinstance(self.model, torch.nn.DataParallel):
                    loss = self.loss_wrapper(output, meta, **self.loss_args)
                    loss = loss.mean()
                else:
                    loss = self.loss(output, meta, **self.loss_args)

                avg_val_loss_trainbn.update(loss.item(), data.size(0))

        for md in self.model.modules():
            if isinstance(md, _BatchNorm):
                md.track_running_stats = True

        val_log = {
            'val_loss': avg_val_loss.avg,
            'val_loss_trainbn': avg_val_loss_trainbn.avg,
            'val_metrics': [a.avg for a in total_val_metrics]
        }

        self.writer.set_step(epoch, 'val_epoch')
        self.writer.add_scalar('val_loss', val_log['val_loss'])
        self.writer.add_scalar('val_loss_trainbn', val_log['val_loss_trainbn'])

        for i, metric in enumerate(self.metrics):
            self.writer.add_scalar(f'{metric.__name__}', val_log['val_metrics'][i])

        torch.set_rng_state(cached_state)

        return val_log
