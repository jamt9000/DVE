import numpy as np
import torch
import time
from torchvision.utils import make_grid
from base import BaseTrainer


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
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None, visualizations=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.visualizations = visualizations if visualizations is not None else []

        class LossWrapper(torch.nn.Module):
            def __init__(self, fn):
                super(LossWrapper, self).__init__()
                self.fn = fn
            def __call__(self, *a, **kw):
                return self.fn(*a,**kw)

        self.loss_wrapper = LossWrapper(self.loss)

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
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
        total_metrics = np.zeros(len(self.metrics))
        seen_tic = time.time()
        seen = 0
        for batch_idx, (data, meta) in enumerate(self.data_loader):
            timings = {}
            batch_tic = time.time()
            data = data.to(self.device)
            seen += data.shape[0]
            timings["data transfer"] = time.time() - batch_tic

            tic = time.time()
            self.optimizer.zero_grad()
            output = self.model(data)
            timings["fwd"] = time.time() - tic

            tic = time.time()
            if isinstance(self.model, torch.nn.DataParallel):
                loss = torch.nn.DataParallel(self.loss_wrapper, device_ids=self.model.device_ids)(output, meta)
                loss = loss.mean()
            else:
                loss = self.loss(output, meta)
            timings["loss-fwd"] = time.time() - tic

            tic = time.time()
            loss.backward()
            timings["loss-back"] = time.time() - tic

            tic = time.time()
            self.optimizer.step()
            timings["optim-step"] = time.time() - tic

            tic = time.time()
            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            avg_loss.update(loss.item(), data.size(0))
            total_metrics += self._eval_metrics(output, meta)
            timings["metrics"] = time.time() - tic

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                toc = time.time() - seen_tic
                tic = time.time()
                msg ='Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} Hz: {:.2f}'
                self.logger.info(msg.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    len(self.data_loader.dataset),
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item(),
                    seen / max(toc, 1E-5)))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                for v in self.visualizations:
                    v(self.writer, data.cpu(), output)
                seen_tic = time.time()
                seen = 0
                timings["vis"] = time.time() - tic

            timings["minibatch"] = time.time() - batch_tic

            print("==============")
            for key in timings:
                ratio = 100 * timings[key] / timings["minibatch"]
                print("{:.3f} ({:.2f}%) >>> {}".format(timings[key], ratio, key))
            print("==============")

        log = {
            'loss': avg_loss.avg,
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        self.writer.set_step(epoch, 'train_epoch')
        self.writer.add_scalar('loss', log['loss'])

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
        self.model.eval()
        avg_val_loss = AverageMeter()
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, meta) in enumerate(self.valid_data_loader):
                data = data.to(self.device)

                output = self.model(data)

                if isinstance(self.model, torch.nn.DataParallel):
                    loss = torch.nn.DataParallel(self.loss_wrapper, device_ids=self.model.device_ids)(output, meta)
                    loss = loss.mean()
                else:
                    loss = self.loss(output, meta)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                avg_val_loss.update(loss.item(), data.size(0))
                total_val_metrics += self._eval_metrics(output, meta)
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                for v in self.visualizations:
                    v(self.writer, data.cpu(), output)

        val_log = {
            'val_loss': avg_val_loss.avg,
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

        self.writer.set_step(epoch, 'val_epoch')
        self.writer.add_scalar('val_loss', val_log['val_loss'])

        return val_log
