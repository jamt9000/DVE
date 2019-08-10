import time
import argparse
import numpy as np
import random
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import utils.visualization as module_visualization
from trainer import Trainer
from utils import Logger, dict_coll
from utils import tps, clean_state_dict, coll, NoGradWrapper, Up, get_instance
from test_matching import evaluation
import torch.nn as nn
from parse_config import ConfigParser
from torch.utils.data import DataLoader
import torch.utils.data.dataloader


def main(config, resume):
    logger = config.get_logger('train')
    seeds = [int(x) for x in config._args.seeds.split(",")]
    torch.backends.cudnn.benchmark = True
    logger.info("Launching experiment with config:")
    logger.info(config)

    for seed in seeds:
        tic = time.time()
        logger.info(f"Setting experiment random seed to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        model = get_instance(module_arch, 'arch', config)
        logger.info(model)

        if 'finetune_from' in config.keys():
            checkpoint = torch.load(config['finetune_from'])
            model.load_state_dict(clean_state_dict(checkpoint["state_dict"]))
            print('Finetuning from %s' % config['finetune_from'])

        if 'keypoint_regressor' in config.keys():
            descdim = config['arch']['args']['num_output_channels']
            kp_regressor = get_instance(module_arch, 'keypoint_regressor', config,
                                        descriptor_dimension=descdim)
            basemodel = NoGradWrapper(model)

            if config.get('keypoint_regressor_upsample', False):
                model = nn.Sequential(basemodel, Up(), kp_regressor)
            else:
                model = nn.Sequential(basemodel, kp_regressor)

        if 'segmentation_head' in config.keys():
            descdim = config['arch']['args']['num_output_channels']
            segmenter = get_instance(module_arch, 'segmentation_head', config,
                                    descriptor_dimension=descdim)
            if config["segmentation_head"]["args"].get("freeze_base", True):
                basemodel = NoGradWrapper(model)
            else:
                basemodel = model

            if config.get('segmentation_upsample', False):
                model = nn.Sequential(basemodel, Up(), segmenter)
            else:
                model = nn.Sequential(basemodel, segmenter)

        # setup data_loader instances
        imwidth = config['dataset']['args']['imwidth']
        warper = get_instance(tps, 'warper', config, imwidth,
                            imwidth) if 'warper' in config.keys() else None

        loader_kwargs = {}
        coll_func = config.get("collate_fn", "dict_flatten")
        if coll_func == "flatten":
            loader_kwargs["collate_fn"] = coll
        elif coll_func == "dict_flatten":
            loader_kwargs["collate_fn"] = dict_coll
        else:
            raise ValueError("collate function type {} unrecognised".format(coll_func))

        dataset = get_instance(module_data, 'dataset', config, pair_warper=warper,
                            train=True)
        if config["disable_workers"]:
            num_workers = 0
        else:
            num_workers = 4

        data_loader = DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            **loader_kwargs,
        )

        warp_val = config.get('warp_val', True)
        val_dataset = get_instance(
            module_data,
            'dataset',
            config,
            train=False,
            pair_warper=warper if warp_val else None,
        )
        valid_data_loader = DataLoader(val_dataset, batch_size=32, **loader_kwargs)

        # get function handles of loss and metrics
        loss = getattr(module_loss, config['loss'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]
        if not config["vis"]:
            visualizations = []
        else:
            visualizations = [
                getattr(module_visualization, vis) for vis in config['visualizations']
            ]

        # build optimizer, learning rate scheduler. delete every lines containing
        # lr_scheduler for disabling scheduler
        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))

        if 'keypoint_regressor' in config.keys():
            base_params = list(filter(lambda p: p.requires_grad, basemodel.parameters()))
            trainable_params = [
                x for x in trainable_params if not sum([(x is w) for w in base_params])
            ]

        biases = [x.bias for x in model.modules() if isinstance(x, nn.Conv2d)]

        trainbiases = [x for x in trainable_params if sum([(x is b) for b in biases])]
        trainweights = [x for x in trainable_params if not sum([(x is b) for b in biases])]
        print(len(trainbiases), 'Biases', len(trainweights), 'Weights')

        bias_lr = config.get('bias_lr', None)
        if bias_lr is not None:
            optimizer = get_instance(torch.optim, 'optimizer', config, [{
                "params": trainweights
            }, {
                "params": trainbiases,
                "lr": bias_lr
            }])
        else:
            optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)

        lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config,
                                    optimizer)
        trainer = Trainer(
            model=model,
            loss=loss,
            metrics=metrics,
            resume=resume,
            config=config,
            optimizer=optimizer,
            data_loader=data_loader,
            lr_scheduler=lr_scheduler,
            visualizations=visualizations,
            mini_train=config._args.mini_train,
            valid_data_loader=valid_data_loader,
        )
        trainer.train()
        duration = time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - tic))
        logger.info(f"Training took {duration}")
        if "keypoint_regressor" not in config.keys():
            config._args.resume = config.save_dir / "model_best.pth"
            config["mini_eval"] = config._args.mini_train
            evaluation(config, logger=logger)
            logger.info(f"Log written to {config.log_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-f', '--folded_correlation',
                        help='whether to use folded correlation (reduce mem)')
    parser.add_argument('-p', '--profile', action="store_true",
                        help='whether to print out profiling information')
    parser.add_argument('-b', '--batch_size', default=None, type=int,
                        help='the size of each minibatch')
    parser.add_argument('-g', '--n_gpu', default=None, type=int,
                        help='if given, override the numb')
    parser.add_argument('--seeds', default="0", help='random seeds')
    parser.add_argument('--mini_train', action="store_true")
    parser.add_argument('--train_single_epoch', action="store_true")
    parser.add_argument('--disable_workers', action="store_true")
    parser.add_argument('--check_bn_working', action="store_true")
    parser.add_argument('--vis', action="store_true")
    config = ConfigParser(parser)

    # if args.config:
    #     # load config file
    #     config = json.load(open(args.config))
    #     path = os.path.join(config['trainer']['save_dir'], config['name'])
    # elif args.resume:
    #     # load config file from checkpoint, in case new config file is not given.
    #     # Use '--config' and '--resume' arguments together to load trained model and
    #     # train more with changed config.
    #     config = torch.load(args.resume)['config']
    # else:
    #     raise AssertionError("config file needs to be specified. Add '-c config.json'")

    # We allow a small number of cmd-line overrides for fast dev
    args = config._args
    if args.folded_correlation is not None:
        config["loss_args"]["fold_corr"] = args.folded_correlation
    if config._args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if config._args.n_gpu is not None:
        config["n_gpu"] = args.n_gpu
    config["profile"] = args.profile
    config["vis"] = args.vis
    config["disable_workers"] = args.disable_workers
    config["trainer"]["check_bn_working"] = args.check_bn_working

    if args.train_single_epoch:
        print("Restring training to a single epoch....")
        config["trainer"]["epochs"] = 1
        config["trainer"]["save_period"] = 1

    main(config, args.resume)
