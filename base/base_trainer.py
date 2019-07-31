import os
import math
import torch
from utils.visualization import WriterTensorboardX


class BaseTrainer:
    """ Base class for all trainers
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config):

        self.config = config
        self.logger = config.get_logger(
            'trainer', config['trainer']['verbosity'])

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            print("Using DataParallel for loss")
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.verbosity = cfg_trainer['verbosity']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = math.inf if self.mnt_mode == 'min' else -math.inf
            self.early_stop = cfg_trainer.get('early_stop', math.inf)

        self.start_epoch = 1

        # setup directory for checkpoint saving
        # if resume:
        #     start_time = os.path.split(os.path.split(resume)[0])[1]
        # else:
        #     start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        # writer_dir = os.path.join(cfg_trainer['log_dir'], config['name'], start_time)
        self.writer = WriterTensorboardX(
            config.log_dir, self.logger, cfg_trainer['tensorboardX'])

        if resume:
            self._resume_checkpoint(resume)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}"
                                ", but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update(
                        {mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({
                        'val_' + mtr.__name__: value[i] for i,
                        mtr in enumerate(self.metrics)
                    })
                else:
                    log[key] = value

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))
            # if self.logger is not None:
            #     self.logger.add_entry(log)
            #     if self.verbosity >= 1:
            #         for key, value in log.items():
            #             self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best
            # checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to
                    # specified metric(mnt_metric)
                    lower = log[self.mnt_metric] <= self.mnt_best
                    higher = log[self.mnt_metric] >= self.mnt_best
                    improved = (self.mnt_mode == 'min' and lower) or \
                               (self.mnt_mode == 'max' and higher)
                except KeyError:
                    msg = "Warning: Metric '{}' not found, perf monitoring is disabled."
                    self.logger.warning(msg.format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False
                    not_improved_count = 0

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Val performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        self.writer.writer.close()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'logger': self.logger,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = os.path.join(
            self.checkpoint_dir,
            'checkpoint-epoch{}.pth'.format(epoch)
        )
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: {} ...".format('model_best.pth'))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            msg = ("Warning: Architecture configuration given in config file is"
                   " different from that of checkpoint."
                   " This may yield an exception while state_dict is being loaded.")
            self.logger.warning(msg)
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        ckpt_opt_type = checkpoint['config']['optimizer']['type']
        if ckpt_opt_type != self.config['optimizer']['type']:
            msg = ("Warning: Optimizer type given in config file is different from"
                   "that of checkpoint.  Optimizer parameters not being resumed.")
            self.logger.warning(msg)
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger = checkpoint['logger']
        msg = "Checkpoint '{}' (epoch {}) loaded"
        self.logger.info(msg .format(resume_path, self.start_epoch))
