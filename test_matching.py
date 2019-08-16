"""
python test_matching.py \
    --config=configs/warp_ims_smallnet_mafl_64d_dve_128in_keypoints-ep57.json \
    --dense_match \
    --device=3
"""
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
from parse_config import ConfigParser
import model.model as module_arch
from utils import tps, clean_state_dict, get_instance
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.visualization import norm_range
import torch.nn.functional as F
from utils.util import dict_coll
from utils.tps import spatial_grid_unnormalized, tps_grid
try:
    from tensorboardX import SummaryWriter
except:
    pass

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # NOQA

sys.path.insert(0, str(Path.home() / "coding/src/zsvision/python"))
try:
    from zsvision.zs_iterm import zs_dispFig # NOQA
except:
    zs_dispFig = lambda: None


def compute_pixel_err(pred_x, pred_y, gt_x, gt_y, imwidth, crop):
    """Compute the pixel error of the corresponding keypoints

    Args:
        pred_x (float): predicted x-coordinate for keypoint
        pred_y (float): predicted y-coordinate for keypoint
        gt_x (float): ground truth x-coordinate for keypoint
        gt_y (float): ground truth y-coordinate for keypoint
        imwidth (int): the width of the image (pixels)
        crop (int): the size of the crop from the boundary (pixels)

    Returns:
        (float) pixel error
    NOTE: To account for different input sizes, we scale all distances as
    though they occured in pixel space for a 70x70 (post-crop) image
    (this was used in the original version of the model so allows
    for comparison).
    """
    canonical_sz = 70
    scale = canonical_sz / (imwidth - 2 * crop)
    pred_x = pred_x * scale
    pred_y = pred_y * scale
    gt_x = gt_x * scale
    gt_y = gt_y * scale
    return np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)


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


def evaluation(config, logger=None, eval_data=None):
    device = torch.device('cuda:0' if config["n_gpu"] > 0 else 'cpu')

    if logger is None:
        logger = config.get_logger('test')

    logger.info("Running evaluation with configuration:")
    logger.info(config)

    imwidth = config['dataset']['args']['imwidth']
    root = config["dataset"]["args"]["root"]
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
    if eval_data is None:
        eval_data = config["dataset"]["type"]
    constructor = getattr(module_data, eval_data)

    # handle the case of the MAFL split, which by default will evaluate on Celeba
    kwargs = {"val_split": "mafl"} if eval_data == "CelebAPrunedAligned_MAFLVal" else {}
    val_dataset = constructor(
        train=False,
        pair_warper=warper,
        use_keypoints=True,
        imwidth=imwidth,
        crop=crop,
        root=root,
        **kwargs,
    )
    # NOTE: Since the matching is performed with pairs, we fix the ordering and then
    # use all pairs for datasets with even numbers of images, and all but one for
    # datasets that have odd numbers of images (via drop_last=True)
    data_loader = DataLoader(val_dataset, batch_size=2, collate_fn=dict_coll,
                             shuffle=False, drop_last=True)

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

    if dense_match:
        warp_dir = Path(config["warp_dir"]) / config["name"]
        warp_dir = warp_dir / "disable_warps{}".format(disable_warps)
        if not warp_dir.exists():
            warp_dir.mkdir(exist_ok=True, parents=True)
        writer = SummaryWriter(warp_dir)

    model.eval()
    same_errs = []
    diff_errs = []

    torch.manual_seed(0)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            data, meta = batch["data"], batch["meta"]

            if (config.get("mini_eval", False) and i > 3):
                break
            # if i == 0:
            #     # Checksum to make sure warps are deterministic
            #     if True:
            #         # redo later
            #         if data.shape[2] == 64:
            #             assert float(data.sum()) == -553.9221801757812
            #         elif data.shape[2] == 128:
            #             assert float(data.sum()) == 754.1907348632812

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

            if config.get("vis", False):
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
                    gt_same_x, gt_same_y = np.array(kp_same[ki])
                    gt_diff_x, gt_diff_y = np.array(kp_diff[ki])
                    same_x, same_y = find_descriptor(x, y, fsrc, fsame, stride)

                    err = compute_pixel_err(
                        pred_x=same_x,
                        pred_y=same_y,
                        gt_x=gt_same_x,
                        gt_y=gt_same_y,
                        imwidth=imwidth,
                        crop=crop,
                    )
                    same_errs.append(err)
                    diff_x, diff_y = find_descriptor(x, y, fsrc, fdiff, stride)
                    err = compute_pixel_err(
                        pred_x=diff_x,
                        pred_y=diff_y,
                        gt_x=gt_diff_x,
                        gt_y=gt_diff_y,
                        imwidth=imwidth,
                        crop=crop,
                    )
                    diff_errs.append(err)
                    if config.get("vis", False):
                        ax2.scatter(same_x, same_y, c='b')
                        ax3.scatter(diff_x, diff_y, c='b')

            if config.get("vis", False):
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
    parser.add_argument('--eval_data', default="MAFLAligned")
    eval_config = ConfigParser(parser)

    eval_config["dense_match"] = eval_config._args.dense_match
    eval_config["vis"] = eval_config._args.vis
    eval_config["mini_eval"] = eval_config._args.mini_eval
    msg = "For evaluation, a model checkpoint must be specified via the --resume flag"
    assert eval_config._args.resume, msg
    evaluation(eval_config, eval_data=eval_config._args.eval_data)
