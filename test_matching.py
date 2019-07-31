"""
python test_matching.py \
    --config=configs/warp_ims_smallnet_mafl_64d_evc_128in_keypoints-ep57.json \
    --dense_match \
    --device=3
"""
import os
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
from parse_config import ConfigParser
import model.model as module_arch
from train import get_instance
from utils import tps, clean_state_dict
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.visualization import norm_range
import torch.nn.functional as F
from utils.util import dict_coll
from utils.tps import spatial_grid_unnormalized, tps_grid
from tensorboardX import SummaryWriter

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # NOQA

sys.path.insert(0, str(Path.home() / "coding/src/zsvision/python"))
from zsvision.zs_iterm import zs_dispFig # NOQA


def find_descriptor(x, y, source_descs, target_descs, stride):
    C, H, W = source_descs.shape
    x = int(np.round(x / stride))
    y = int(np.round(y / stride))
    x = min(W - 1, max(x, 0))
    y = min(H - 1, max(y, 0))
    query_desc = source_descs[:, y, x]
    corr = torch.matmul(query_desc.reshape(-1, C), target_descs.reshape(C, H * W))
    maxidx = corr.argmax()
    grid = spatial_grid_unnormalized(H, W).reshape(-1, 2) * stride
    x, y = grid[maxidx]
    return x.item(), y.item()


def dense_desc_match(src, target, upscale=2):

    # upsample for higher resolution
    interp_kwargs = dict(scale_factor=upscale, mode='bilinear', align_corners=True)
    src = F.interpolate(src.unsqueeze(0), **interp_kwargs).squeeze(0)
    target = F.interpolate(target.unsqueeze(0), **interp_kwargs).squeeze(0)
    C, H, W = src.shape
    # target = F.interpolate(target.unsqueeze(0), **interp_kwargs).squeeze(0)
    grid = tps_grid(H, W)
    # to (H x W x H x W)
    corr = torch.einsum("ijk,ilm->jklm", src, target)
    # corr2 = torch.matmul(
    #     source_descs.permute(1, 2, 0).reshape(-1, C),
    #     target_descs.reshape(C, H * W),
    # )
    # corr2 = corr2.reshape(H, W, H, W)
    # find maximal correlation among source
    maxidx = torch.argmax(corr.view(H * W, H * W), dim=0)
    return grid[maxidx].reshape(1, H, W, 2)
    # return picks


def evaluation(config, logger=None):
    device = torch.device('cuda:0' if config["n_gpu"] > 0 else 'cpu')

    if logger is None:
        logger = config.get_logger('test')

    logger.info("Running evaluation with configuration:")
    logger.info(config)

    imwidth = config['dataset']['args']['imwidth']
    warp_crop_default = config['warper']['args'].get('crop', None)
    crop = config['dataset']['args'].get('crop', warp_crop_default)

    # Want explicit pair warper
    disable_warps = True
    dense_match = config.get("dense_match", False)
    if dense_match and disable_warps:
        # rotsd = 2.5
        # scalesd=0.1 * .5
        rotsd = 0
        scalesd = 0
        warp_kwargs = dict(
            warpsd_all=0,
            warpsd_subset=0,
            transsd=0,
            scalesd=scalesd,
            rotsd=rotsd,
            im1_multiplier=1,
            im1_multiplier_aff=1
        )
    else:
        warp_kwargs = dict(
            warpsd_all=0.001 * .5,
            warpsd_subset=0.01 * .5,
            transsd=0.1 * .5,
            scalesd=0.1 * .5,
            rotsd=5 * .5,
            im1_multiplier=1,
            im1_multiplier_aff=1
        )
    warper = tps.Warper(imwidth, imwidth, **warp_kwargs)
    warper1 = tps.WarperSingle(
        imwidth,
        imwidth,
        warpsd_all=0.0,
        warpsd_subset=0.0,
        transsd=0.05,
        scalesd=0.01,
        rotsd=2
    )

    if False:
        train_dataset = module_data.MAFLAligned(
            root='data/celeba',
            imwidth=imwidth,
            crop=crop,
            train=True,
            pair_warper=warper1,
            do_augmentations=False
        )
        val_dataset = module_data.MAFLAligned(
            root='data/celeba',
            imwidth=imwidth,
            crop=crop,
            train=False,
            pair_warper=warper,
            use_keypoints=True
        )
    else:
        train_dataset = get_instance(
            module_data,
            'dataset',
            config,
            pair_warper=warper1,
            train=True,
            use_keypoints=True,
        )
        val_dataset = get_instance(
            module_data,
            'dataset',
            config,
            pair_warper=warper,
            train=False,
            use_keypoints=True,
        )

    val_batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=val_batch_size, shuffle=False)
    data_loader = DataLoader(val_dataset, batch_size=2, collate_fn=dict_coll,
                             shuffle=False)

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    model.summary()

    # load state dict
    ckpt_path = config._args.resume
    logger.info(f"Loading checkpoint: {ckpt_path} ...")
    checkpoint = torch.load(ckpt_path)
    # checkpoint = torch.load(config["weights"])
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(clean_state_dict(state_dict))
    if config['n_gpu'] > 1:
        model = model.module

    model = model.to(device)
    model.train()

    # Stabilise batch norm
    reinit_bn = False
    if reinit_bn:
        torch.manual_seed(0)
        bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        for m in bns:
            m.reset_running_stats()

        train_loader_it = iter(train_loader)
        with torch.no_grad():
            for i in range(100):
                torch.manual_seed(0)
                np.random.seed(0)

                data, meta = next(train_loader_it)
                data = data.to(device)
                print(i, 'data checksum', float(data.sum()))
                output = model(data)
                print(i, 'bn checksum', float(bns[0].running_mean.sum()))

    if dense_match:
        warp_dir = Path(config["warp_dir"]) / config["name"]
        warp_dir = warp_dir / "disable_warps{}".format(disable_warps)
        if not warp_dir.exists():
            warp_dir.mkdir(exist_ok=True, parents=True)
        writer = SummaryWriter(warp_dir)

    model.eval()
    same_errs = []
    diff_errs = []
    # Use 200 pairs to reproduce the experiment in the paper
    num_batches = 200 / val_batch_size

    torch.manual_seed(0)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            data, meta = batch["data"], batch["meta"]

            if (config.get("mini_eval", False) and i > 3) or (i > num_batches):
                break

            if i == 0:
                # Checksum to make sure warps are deterministic
                if True:
                    # redo later
                    if data.shape[2] == 64:
                        assert float(data.sum()) == -553.9221801757812
                    elif data.shape[2] == 128:
                        assert float(data.sum()) == 754.1907348632812

            data = data.to(device)
            output = model(data)

            descs = output[0]
            descs1 = descs[0::2]  # 1st in pair (more warped)
            descs2 = descs[1::2]  # 2nd in pair
            ims1 = data[0::2].cpu()
            ims2 = data[1::2].cpu()

            im_source = ims1[0]
            im_same = ims2[0]
            im_diff = ims2[1]

            C, imH, imW = im_source.shape
            B, C, H, W = descs1.shape
            stride = imW / W

            desc_source = descs1[0]
            desc_same = descs2[0]
            desc_diff = descs2[1]

            if not dense_match:
                kp1 = meta['kp1']
                kp2 = meta['kp2']
                kp_source = kp1[0]
                kp_same = kp2[0]
                kp_diff = kp2[1]

            if config["vis"]:
                fig = plt.figure()  # a new figure window
                ax1 = fig.add_subplot(1, 3, 1)
                ax2 = fig.add_subplot(1, 3, 2)
                ax3 = fig.add_subplot(1, 3, 3)

                ax1.imshow(norm_range(im_source).permute(1, 2, 0))
                ax2.imshow(norm_range(im_same).permute(1, 2, 0))
                ax3.imshow(norm_range(im_diff).permute(1, 2, 0))

                if not dense_match:
                    ax1.scatter(kp_source[:, 0], kp_source[:, 1], c='g')
                    ax2.scatter(kp_same[:, 0], kp_same[:, 1], c='g')
                    ax3.scatter(kp_diff[:, 0], kp_diff[:, 1], c='g')

            if False:
                fsrc = F.normalize(desc_source, p=2, dim=0)
                fsame = F.normalize(desc_same, p=2, dim=0)
                fdiff = F.normalize(desc_diff, p=2, dim=0)
            else:
                fsrc = desc_source.clone()
                fsame = desc_same.clone()
                fdiff = desc_diff.clone()

            if dense_match:
                # if False:
                #     print("DEBUGGING WITH IDENTICAL FEATS")
                #     fdiff = fsrc
                # tic = time.time()
                grid = dense_desc_match(fsrc, fdiff)
                im_warped = F.grid_sample(im_source.view(1, 3, imH, imW), grid)
                im_warped = im_warped.squeeze(0)
                # print("done matching in {:.3f}s".format(time.time() - tic))
                plt.close("all")
                if config["subplots"]:
                    fig = plt.figure()  # a new figure window
                    ax1 = fig.add_subplot(1, 3, 1)
                    ax2 = fig.add_subplot(1, 3, 2)
                    ax3 = fig.add_subplot(1, 3, 3)
                    ax1.imshow(norm_range(im_source).permute(1, 2, 0))
                    ax2.imshow(norm_range(im_diff).permute(1, 2, 0))
                    ax3.imshow(norm_range(im_warped).permute(1, 2, 0))
                    triplet_dest = warp_dir / "triplet-{:05d}.jpg".format(i)
                    fig.savefig(triplet_dest)
                else:
                    triplet_dest_dir = warp_dir / "triplet-{:05d}".format(i)
                    if not triplet_dest_dir.exists():
                        triplet_dest_dir.mkdir(exist_ok=True, parents=True)
                    for jj, im in enumerate((im_source, im_diff, im_warped)):
                        plt.axis("off")
                        fig = plt.figure(figsize=(1.5, 1.5))
                        ax = plt.Axes(fig, [0., 0., 1., 1.])
                        ax.set_axis_off()
                        fig.add_axes(ax)
                        # ax.imshow(data, cmap = plt.get_cmap("bone"))
                        im_ = norm_range(im).permute(1, 2, 0)
                        ax.imshow(im_)
                        dest_path = triplet_dest_dir / "im-{}-{}.jpg".format(i, jj)
                        plt.savefig(str(dest_path), dpi=im_.shape[0])
                        # plt.savefig(filename, dpi = sizes[0])
                writer.add_figure('warp-triplets', fig)
            else:
                for ki, kp in enumerate(kp_source):
                    x, y = np.array(kp)
                    gt_samex, gt_samey = np.array(kp_same[ki])
                    gt_diffx, gt_diffy = np.array(kp_diff[ki])
                    samex, samey = find_descriptor(x, y, fsrc, fsame, stride)
                    err = np.sqrt((gt_samex - samex)**2 + (gt_samey - samey)**2)
                    same_errs.append(err)
                    diffx, diffy = find_descriptor(x, y, fsrc, fdiff, stride)
                    err = np.sqrt((gt_diffx - diffx)**2 + (gt_diffy - diffy)**2)
                    diff_errs.append(err)
                    if config["vis"]:
                        ax2.scatter(samex, samey, c='b')
                        ax3.scatter(diffx, diffy, c='b')

            if config["vis"]:
                zs_dispFig()
                fig.savefig('/tmp/matching.pdf')

    print("")  # cleanup print from tqdm subtraction
    logger.info("Matching Metrics:")
    logger.info(f"Mean Pixel Error (same-identity): {np.mean(same_errs)}")
    logger.info(f"Mean Pixel Error (different-identity) {np.mean(diff_errs)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--config', help="config file path")
    parser.add_argument('--resume', help='path to ckpt for evaluation')
    parser.add_argument('--device', help='indices of GPUs to enable')
    parser.add_argument('--mini_eval', action="store_true")
    parser.add_argument('--vis', action="store_true")
    parser.add_argument('--dense_match', action="store_true")
    parser.add_argument('--subplots', action="store_true")
    eval_config = ConfigParser(parser)

    eval_config["dense_match"] = eval_config._args.dense_match
    eval_config["vis"] = eval_config._args.vis
    eval_config["mini_eval"] = eval_config._args.mini_eval
    msg = "For evaluation, a model checkpoint must be specified via the --resume flag"
    assert eval_config._args.resume, msg
    evaluation(eval_config)