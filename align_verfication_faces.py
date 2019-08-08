"""
python align_verification_faces.py \
    --config=configs/warp_ims_smallnet_mafl_64d_dve_128in_keypoints-ep57.json \
    --device=1 \
    --subplots \
    --sanity_check
"""
import os
import argparse
import torch
from tqdm import tqdm
import time
import data_loader.data_loaders as module_data
import json
import model.model as module_arch
import cv2
from train import get_instance
from utils import clean_state_dict, dict_coll
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.visualization import norm_range
import torch.nn.functional as F
from utils.tps import spatial_grid_unnormalized, tps_grid
from tensorboardX import SummaryWriter
from skimage import transform

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # NOQA

sys.path.insert(0, str(Path.home() / "coding/src/zsvision/python"))
from zsvision.zs_iterm import zs_dispFig # NOQA


def dense_desc_match(src, targets, upscale=2, l2norm=False, similarity=False):
    interp_kwargs = dict(scale_factor=upscale, mode='bilinear', align_corners=True)
    src = F.interpolate(src.unsqueeze(0), **interp_kwargs).squeeze(0)
    targets = F.interpolate(targets, **interp_kwargs)
    C, H, W = src.shape
    grid = tps_grid(H, W)
    if l2norm:
        src = F.normalize(src, p=2, dim=0)
        targets = F.normalize(targets, p=2, dim=1)
    # corr = torch.einsum("ijk,ilm->jklm", src, target)
    corr = torch.einsum("jkl,ijmn->iklmn", src, targets)
    num_targets = targets.shape[0]
    corr = corr.view(H * W, num_targets, H * W)
    # find best match among src
    corr = corr.max(dim=1)[0]
    maxidx = torch.argmax(corr.view(H * W, H * W), dim=0)
    grid = grid[maxidx].reshape(1, H, W, 2)

    if similarity:
        base_grid = tps_grid(H, W)
        params = fit_similiarity(src=grid, target=base_grid)
        # upscale to maintain high resolution in images
        assert upscale == 1, "expected to only upscale after fitting tx"
        H_, W_ = 2 * H, 2 * W
        base_grid = tps_grid(H_, W_).to(src.device)
        ones = torch.ones(base_grid.size(0), 1, device=src.device)
        base_grid = torch.cat(tensors=(base_grid, ones), dim=1)
        pred_grid = torch.matmul(params.to(src.device), base_grid.t())[:2].t()
        grid = pred_grid.view(1, H_, W_, 2)
    return grid


def fit_similiarity(src, target):
    src = src.view(-1, 2)
    target = target.view(-1, 2)
    tform = transform.SimilarityTransform()
    tform.estimate(target.cpu().numpy(), src.cpu().numpy())
    # ones = torch.ones(target.size(0), 1, device=target.device)
    # target = torch.cat(tensors=(target, ones), dim=1)
    return torch.from_numpy(tform.params).to(src.device).float()


def main(config):
    device = 'cuda'
    imwidth = config['dataset']['args']['imwidth']
    crop = config["dataset"]["args"]["crop"]
    data_kwargs = {"root": "data/ijbb", "train": False, "imwidth": imwidth - 2 * crop}
    proto_dataset = module_data.IJBB(prototypes=True, **data_kwargs)
    val_dataset = module_data.IJBB(prototypes=False, **data_kwargs)
    loader_kwargs = {"batch_size": 1, "shuffle": False, "collate_fn": dict_coll,
                     "num_workers": 2}
    proto_loader = DataLoader(proto_dataset, **loader_kwargs)
    data_loader = DataLoader(val_dataset, **loader_kwargs)

    model = get_instance(module_arch, 'arch', config)
    # model.summary()

    checkpoint = torch.load(config["weights"])
    model.load_state_dict(clean_state_dict(checkpoint["state_dict"]))
    model = model.to(device)

    model.eval()
    torch.manual_seed(0)
    protos = {}
    if config["similarity"]:
        upscale = 1
    else:
        upscale = 2
    match_kwargs = {
        "upscale": upscale,
        "l2norm": config["l2norm"],
        "similarity": config["similarity"],
    }

    print("storing prototype features")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(proto_loader)):
            data, im_paths = batch["data"], batch["im_path"]
            data = data.to(device)
            output = model(data)[0]
            assert len(im_paths) == 1, "expected single element batches"
            protos[im_paths[0]] = {"im": data[0], "feats": output}

    if config["sanity_check"]:
        print("sanity checking prototype features")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(proto_loader)):
                data, im_paths = batch["data"], batch["im_path"]
                data = data.to(device)
                fsrc = model(data)[0][0]
                im_source = data[0]
                C, imH, imW = im_source.shape

                for jj, (proto_im_path, proto) in enumerate(protos.items()):
                    fproto, im_proto = proto["feats"], proto["im"]
                    grid = dense_desc_match(fsrc, fproto, **match_kwargs).to(device)
                    # if config["similarity"]:
                    #     grid = grid.squeeze(0)
                    #     H, W = grid.shape[0], grid.shape[1]
                    #     base_grid = tps_grid(H, W)
                    #     pred_grid = solve_for_similiarity(grid, base_grid)
                    #     grid = pred_grid.view(1, H, W, 2)
                    im_warped = F.grid_sample(im_source.view(1, 3, imH, imW), grid)
                    im_warped = im_warped.squeeze(0).cpu()
                    plt.close("all")
                    if config["subplots"]:
                        fig = plt.figure()
                        ax1 = fig.add_subplot(1, 3, 1)
                        ax2 = fig.add_subplot(1, 3, 2)
                        ax3 = fig.add_subplot(1, 3, 3)
                        ax1.imshow(norm_range(im_source).permute(1, 2, 0).cpu())
                        ax2.imshow(norm_range(im_proto).permute(1, 2, 0).cpu())
                        ax3.imshow(norm_range(im_warped).permute(1, 2, 0).cpu())
                        msg = "{}/{} - aligning to {}"
                        msg = msg.format(jj, len(protos), Path(proto_im_path).name)
                        plt.suptitle(msg)
                        zs_dispFig()
                        import ipdb; ipdb.set_trace()

    num_protos = config["num_protos"]
    print("warping to nearest prototype of {}".format(num_protos))
    protos = {k: v for ii, (k, v) in enumerate(protos.items()) if ii < num_protos}
    assert len(protos) == 1, "expected single prototype"
    import ipdb; ipdb.set_trace()

    proto_stem = "proto-{}".format(Path(list(protos.keys())[0]).stem)
    exp_name = "{}-l2norm{}-{}".format(config["name"], config["l2norm"], proto_stem)
    if config["similarity"]:
        exp_name += "-similiarity"
    warp_dir = Path(config["warp_dir"]) / exp_name
    align_dir = Path(config["align_dir"]) / exp_name
    if not warp_dir.exists():
        warp_dir.mkdir(exist_ok=True, parents=True)
    if not align_dir.exists():
        align_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(warp_dir)

    # if not config["subplots"]:
    #     # shared axis to avoid slowdown
    #     plt.close("all")
    #     plt.axis("off")
    #     fig = plt.figure(figsize=(1, 1))
    #     ax = plt.Axes(fig, [0., 0., 1., 1.])
    #     ax.set_axis_off()
    #     fig.add_axes(ax)
    #     h = None

    with torch.no_grad():
        load_tic = time.time()
        for i, batch in enumerate(tqdm(data_loader)):
            data, im_paths = batch["data"], batch["im_path"]
            dest_path = align_dir / Path(im_paths[0]).name
            if dest_path.exists():
                continue
            data = data.to(device)

            if config["profile"]:
                print("load time: {:.3f}".format(time.time() - load_tic))
            fsrc = model(data)[0][0]
            if config["profile"]:
                print("fwd time: {:.3f}".format(time.time() - load_tic))

            im_source = data[0]
            C, imH, imW = im_source.shape

            for jj, (proto_im_path, proto) in enumerate(protos.items()):
                fproto, im_proto = proto["feats"], proto["im"]

                if config["profile"]:
                    tic = time.time()
                grid = dense_desc_match(fsrc, fproto, **match_kwargs).to(device)
                if config["profile"]:
                    print("matching time: {:.3f}".format(time.time() - tic))
                    tic = time.time()
                # if config["similarity"]:
                #     grid = grid.squeeze(0)
                #     H, W = grid.shape[0], grid.shape[1]
                #     base_grid = tps_grid(H, W)
                #     pred_grid = solve_for_similiarity(grid, base_grid)
                #     grid = pred_grid.view(1, H, W, 2)
                # if config["profile"]:
                #     print("sim solving time: {:.3f}".format(time.time() - tic))
                #     tic = time.time()
                im_warped = F.grid_sample(im_source.view(1, 3, imH, imW), grid)
                im_warped = im_warped.squeeze(0).cpu()
                if config["profile"]:
                    print("warp time: {:.3f}".format(time.time() - tic))
                    tic = time.time()
                # print("done matching in {:.3f}s".format(time.time() - tic))
                if config["subplots"]:
                    fig = plt.figure()  # a new figure window
                    ax1 = fig.add_subplot(1, 3, 1)
                    ax2 = fig.add_subplot(1, 3, 2)
                    ax3 = fig.add_subplot(1, 3, 3)
                    ax1.imshow(norm_range(im_source).permute(1, 2, 0).cpu())
                    ax2.imshow(norm_range(im_proto).permute(1, 2, 0).cpu())
                    ax3.imshow(norm_range(im_warped).permute(1, 2, 0).cpu())
                    msg = "{}/{} - aligning to {}"
                    plt.suptitle(msg.format(jj, len(protos), Path(proto_im_path).name))
                    # triplet_dest = warp_dir / "triplet-{:05d}.jpg".format(i)
                    # fig.savefig(triplet_dest)
                    writer.add_figure('warp-triplets', fig, global_step=i)
                else:
                    # ax.imshow(data, cmap = plt.get_cmap("bone"))
                    # plt.cla()
                    im_ = norm_range(im_warped)
                    im_ = (im_ * 255).permute(1, 2, 0).cpu().numpy()
                    # switch BGR/RGB
                    im_ = np.round(im_).astype(np.uint8)[:, :, ::-1]
                    # if i > 0:
                    #     h.set_data(im_)
                    # else:
                    # ax = plt.gca()
                    # h = ax.imshow(im_)
                    # import ipdb; ipdb.set_trace()
                    # print("NO SAVE")
                    cv2.imwrite(str(dest_path), im_)
                    # plt.savefig(str(dest_path), dpi=im_.shape[0])
                if config["profile"]:
                    print("fig time: {:.3f}".format(time.time() - tic))
                    load_tic = time.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--device', type=str, help='indices of GPUs to enable')
    parser.add_argument('--vis', action="store_true")
    parser.add_argument('--l2norm', action="store_true")
    parser.add_argument('--subplots', action="store_true")
    parser.add_argument('--num_protos', type=int, default=1)
    parser.add_argument('--sanity_check', action="store_true")
    parser.add_argument('--profile', action="store_true")
    parser.add_argument('--similarity', action="store_true")
    args = parser.parse_args()

    if args.config:
        config = json.load(open(args.config))
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    config.update(vars(args))
    main(config)
