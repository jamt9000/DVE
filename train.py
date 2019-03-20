import os
import json
import argparse
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import utils.visualization as module_visualization
from trainer import Trainer
from utils import Logger
from utils import tps
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data.dataloader


def get_instance(module, name, config, *args, **kwargs):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'], **kwargs)


def coll(batch):
    b = torch.utils.data.dataloader.default_collate(batch)
    # Flatten to be 4D
    return [bi.reshape((-1,) + bi.shape[-3:]) if isinstance(bi, torch.Tensor) else bi for bi in b]


class NoGradWrapper(nn.Module):
    def __init__(self, wrapped):
        super(NoGradWrapper, self).__init__()
        self.wrapped_module = wrapped

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.wrapped_module.forward(*args, **kwargs)


def main(config, resume):
    train_logger = Logger()

    # setup data_loader instances
    imwidth = config['dataset']['args']['imwidth']
    warper = get_instance(tps, 'warper', config, imwidth, imwidth) if 'warper' in config.keys() else None
    dataset = get_instance(module_data, 'dataset', config, pair_warper=warper)
    data_loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        num_workers=max(8, int(config['n_gpu']) * 4),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=coll,
    )

    warp_val = config.get('warp_val', True)
    val_dataset = get_instance(
        module_data,
        'dataset',
        config,
        train=False,
        pair_warper=warper if warp_val else None,
    )
    valid_data_loader = DataLoader(val_dataset, batch_size=32, collate_fn=coll)

    # build model architecture
    model = get_instance(module_arch, 'arch', config)

    if 'finetune_from' in config.keys():
        checkpoint = torch.load(config['finetune_from'])
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k[:7] == 'module.':
                k = k[7:]  # remove `module.`
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        print('Finetuning from %s' % config['finetune_from'])

    if 'keypoint_regressor' in config.keys():
        descdim = config['arch']['args']['num_output_channels']
        kp_regressor = get_instance(module_arch, 'keypoint_regressor', config, descriptor_dimension=descdim)
        basemodel = NoGradWrapper(model)
        model = nn.Sequential(basemodel, kp_regressor)

    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    visualizations = [getattr(module_visualization, vis) for vis in config['visualizations']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))

    biases = [x.bias for x in model.modules() if isinstance(x,nn.Conv2d)]

    trainbiases = [x for x in trainable_params if sum([(x is b) for b in biases])]
    trainweights = [x for x in trainable_params if not sum([(x is b) for b in biases])]
    print(len(trainbiases), 'Biases', len(trainweights), 'Weights')

    bias_lr = config.get('bias_lr', None)
    if bias_lr is not None:
        optimizer = get_instance(torch.optim, 'optimizer', config, [{"params": trainweights}, {"params": trainbiases, "lr": bias_lr}])
    else:
        optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)

    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)

    trainer = Trainer(model, loss, metrics, optimizer,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger,
                      visualizations=visualizations)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-f', '--folded_correlation', default=0, type=int,
                        help='whether to use folded correlation (reduce mem)')
    parser.add_argument('-p', '--profile', default=0, type=int,
                        help='whether to print out profiling information')
    parser.add_argument('-b', '--batch_size', default=None, type=int,
                        help='the size of each minibatch')
    parser.add_argument('-g', '--n_gpu', default=None, type=int,
                        help='if given, override the numb')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")
    config["fold_corr"] = args.folded_correlation
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.n_gpu is not None:
        config["n_gpu"] = args.n_gpu
    config["profile"] = args.profile

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)
