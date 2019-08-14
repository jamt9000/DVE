import sys
import skimage.io
from PIL import Image
import torch
import argparse
import time
import getpass
import model.model as module_arch
from parse_config import ConfigParser
from utils import tps, clean_state_dict, get_instance
from torchvision import transforms
from test_matching import find_descriptor
import tqdm
from data_loader import data_loaders
from utils.visualization import norm_range
import numpy as np
from utils.util import read_json
import os
import matplotlib
from pathlib import Path
from collections import defaultdict
import shelve

# matplotlib.font_manager._rebuild()
matplotlib.rc('font', family='serif', serif='cmr10')
matplotlib.rc('font', monospace='Space Mono')
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.tt'] = 'monospace'
matplotlib.rcParams['lines.markersize'] = 4


if sys.platform == 'darwin':
    matplotlib.use("macosx")
else:
    matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--frame_dir", default="/tmp")
parser.add_argument("--fig_dir", default="data/figs")
parser.add_argument("--canonical_paths", action="store_true")
parser.add_argument("--save_hq_ims", action="store_true")
parser.add_argument("--hq_frame_snapshot", type=int, default=96)
parser.add_argument("--aflw_mtfl_root", default="data",
                    choices=["data", "data/aflw-mtfl"])
args = parser.parse_args()

config_file = 'configs/celeba/smallnet-64d-dve.json'

model_files_nodve = ['data/models/celeba-smallnet-3d/celeba-smallnet-3d/2019-08-04_17-55-48/checkpoint-epoch100.pth',
                     'data/models/celeba-smallnet-16d/celeba-smallnet-16d/2019-08-04_17-55-52/checkpoint-epoch100.pth',
                     'data/models/celeba-smallnet-32d/celeba-smallnet-32d/2019-08-04_17-55-57/checkpoint-epoch100.pth',
                     'data/models/celeba-smallnet-64d/celeba-smallnet-64d/2019-08-04_17-56-04/checkpoint-epoch100.pth']
model_files_dve = [
    'data/models/celeba-smallnet-3d-dve/celeba-smallnet-3d-dve/2019-08-08_17-54-21/checkpoint-epoch100.pth',
    'data/models/celeba-smallnet-16d-dve/celeba-smallnet-16d-dve/2019-08-02_06-20-13/checkpoint-epoch100.pth',
    'data/models/celeba-smallnet-32d-dve/celeba-smallnet-32d-dve/2019-08-02_06-19-59/checkpoint-epoch100.pth',
    'data/models/celeba-smallnet-64d-dve/celeba-smallnet-64d-dve/2019-08-02_06-20-28/checkpoint-epoch100.pth']


if args.canonical_paths:
    def prune_repeated_dir(pp):
        return [Path(x).parents[2] / Path(x).relative_to(Path(x).parents[1]) for x in pp]
    model_files_dve = prune_repeated_dir(model_files_dve)
    model_files_nodve = prune_repeated_dir(model_files_nodve)
    

model_files_all = model_files_nodve + model_files_dve


def grow_axis(ax, d):
    l, b, r, t = ax.get_position().extents
    ax.set_position(matplotlib.transforms.Bbox.from_extents((l - d, b - d, r + d, t + d)))


def load_model_for_eval(checkpoint):
    config_file = Path(checkpoint).parent / 'config.json'
    config = read_json(config_file)
    model = get_instance(module_arch, 'arch', config)
    model.summary()
    checkpoint = torch.load(checkpoint, map_location='cpu')
    state_dict = checkpoint['state_dict']
    model.load_state_dict(clean_state_dict(state_dict))
    model.eval()
    return model


tic = time.time()

avface = skimage.io.imread('https://www.uni-regensburg.de/Fakultaeten/phil_Fak_II/Psychologie/Psy_II/' +
                           'beautycheck/english/durchschnittsgesichter/w(01-64)_gr.jpg')
avface = Image.fromarray(avface)

if args.save_hq_ims:
    from zsvision.zs_iterm import zs_dispFig
    plt.figure(figsize=(15, 15))
    query_ax = plt.subplot(1, 1, 1)
    plt.sca(query_ax)
    # mimic crop
    w, h = avface.size
    cropped = avface.crop((15, 15, w - 15, h - 15))
    plt.imshow(cropped)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig(str(Path(args.fig_dir) / "query-face.png"))
    expected_width, expected_height = 70, 70
    w_crop, h_crop = cropped.size 
    h_ratio = h_crop / expected_height
    w_ratio = w_crop / expected_width
    hq_i_idxs = np.arange(10, 60, 5) * h_ratio
    hq_j_idxs = np.arange(15, 60, 5) * w_ratio
    npts = len(hq_i_idxs) * len(hq_j_idxs)

    rainbow = plt.cm.Spectral(np.linspace(0, 1, npts))
    # plt.xlabel('Query')
    plt.gca().set_prop_cycle('color', rainbow)
    grow_axis(query_ax, -0.05)
    fac = plt.gca().get_position().width
    # fac = plt.gca().get_position().width / dve_ax.get_position().width


    for i in hq_i_idxs:
        for j in hq_j_idxs:
            plt.scatter(j, i, s=(matplotlib.rcParams['lines.markersize']) * 10 ** 2)
    zs_dispFig()
    Path(args.fig_dir).mkdir(exist_ok=True, parents=True)
    plt.savefig(str(Path(args.fig_dir) / "query-face-grid.png"))

imsize = 70
n_images_to_load = 200
dataset = data_loaders.AFLW_MTFL(args.aflw_mtfl_root, train=False, imwidth=imsize)

models_dict = dict([(c, load_model_for_eval(c)) for c in model_files_all])

sample_ims = defaultdict(list)

# Disk backed cache
sample_descs = shelve.open('/tmp/desccache')
sample_descs.clear()
for samplei in range(n_images_to_load):
    for m in model_files_all:
        model = models_dict[m]
        item = dataset[samplei]
        sample_im = item['data']
        sample_desc = model.forward(sample_im.unsqueeze(0))[0][0]

        sample_ims[m].append(sample_im)
        sample_descs[m] = sample_descs.get(m,[]) + [sample_desc]

normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769],
                                 std=[0.2599, 0.2371, 0.2323])
augmentations = []

transforms = transforms.Compose([transforms.Resize((imsize, imsize)), transforms.ToTensor(), normalize])
avface_tensor = transforms(avface)

descs = {}

for m in model_files_all:
    model = models_dict[m]
    avdescs = model.forward(avface_tensor.unsqueeze(0))[0][0]
    descs[m] = avdescs
    imC, imH, imW = avface_tensor.shape
    _, H, W = avdescs.shape
    stride = imW / W

i_idxs = np.arange(10, 60, 5)
j_idxs = np.arange(15, 60, 5)
npts = len(i_idxs) * len(j_idxs)




def nudge_axis(ax, d):
    l, b, r, t = ax.get_position().extents
    ax.set_position(matplotlib.transforms.Bbox.from_extents((l + d, b, r + d, t)))



plt.figure(figsize=(7, 3))

query_ax = plt.subplot(1, 3, 2)
nodve_ax = plt.subplot(1, 3, 1, frameon=False)
dve_ax = plt.subplot(1, 3, 3, frameon=False)

nodve_ax.axis('square')
grow_axis(nodve_ax, 0.05)
nudge_axis(nodve_ax, 0.03)

dve_ax.axis('square')
grow_axis(dve_ax, 0.05)
nudge_axis(dve_ax, -0.03)

plt.sca(query_ax)
plt.imshow(norm_range(avface_tensor).permute(1, 2, 0))
rainbow = plt.cm.Spectral(np.linspace(0, 1, npts))
plt.xlabel('Query')
plt.gca().set_prop_cycle('color', rainbow)
grow_axis(query_ax, -0.05)
plt.xticks([], [])
plt.yticks([], [])

fac = plt.gca().get_position().width / dve_ax.get_position().width

for i in i_idxs:
    for j in j_idxs:
        plt.scatter(j, i, s=(matplotlib.rcParams['lines.markersize'] * fac) ** 2)


def ax_reset():
    plt.cla()
    plt.axis('square')
    plt.xlim(query_ax.get_xlim())
    plt.ylim(query_ax.get_ylim())
    plt.xticks([], [])
    plt.yticks([], [])


def tween_scatter(t, im1, im2, scatter1, scatter2, title1, title2, fade_ims=True,
                  heading1=None, heading2=None, frame=None, is_dve=None):
    ax_reset()
    base_subplot = plt.gca()
    plt.subplot(base_subplot)

    gridsize = int(np.sqrt(len(im1)))

    inner_grid = matplotlib.gridspec.GridSpec(gridsize, gridsize, hspace=0.05, wspace=0.05)
    bb = base_subplot.get_position()
    l, b, r, tp = bb.extents
    inner_grid.update(left=l, bottom=b, right=r, top=tp)

    if fade_ims:
        prev_alpha = np.maximum(0., 1 - 2 * t)
        cur_alpha = np.maximum(0., -1 + 2 * t)
    else:
        prev_alpha = 0.
        cur_alpha = 1.

    for gi in range(gridsize ** 2):
        gax = plt.gcf().add_subplot(inner_grid[gi])

        ax_reset()

        if prev_alpha:
            plt.imshow(norm_range(im1[gi]).permute(1, 2, 0), alpha=prev_alpha)
        if cur_alpha:
            plt.imshow(norm_range(im2[gi]).permute(1, 2, 0), alpha=cur_alpha)

        ease = (-np.cos(np.pi * t) + 1) / 2
        scatter_tween = (1 - ease) * scatter1[gi] + ease * scatter2[gi]

        fac = plt.gca().get_position().width / base_subplot.get_position().width
        plt.scatter(scatter_tween[:, 0], scatter_tween[:, 1], c=rainbow,
                    s=(matplotlib.rcParams['lines.markersize'] * fac) ** 2)

        if frame == args.hq_frame_snapshot and args.save_hq_ims:
            # create temp figure inline
            prev_fig = plt.gcf()
            plt.figure(figsize=(15, 15))
            inline_ax = plt.subplot(1, 1, 1)
            plt.sca(inline_ax)
            plt.imshow(norm_range(im1[gi]).permute(1, 2, 0))
            plt.xticks([], [])
            plt.yticks([], [])
            fname = f"frame{frame}-match-face{gi}"
            if is_dve is not None and is_dve:
                fname += "-dve"
            plt.scatter(scatter2[gi][:, 0], scatter2[gi][:, 1], c=rainbow,
                        s=(matplotlib.rcParams['lines.markersize'] * 8) ** 2)
            plt.savefig(str(Path(args.fig_dir) / f"{fname}.png"))
            zs_dispFig()

            # return to prev figure
            plt.figure(prev_fig.number)

    plt.sca(base_subplot)
    ttl1 = plt.text(0.5, -.08, title1, transform=base_subplot.transAxes, horizontalalignment='center')
    ttl2 = plt.text(0.5, -.08, title2, transform=base_subplot.transAxes, horizontalalignment='center')
    ttl1.set_alpha(1 - t)
    ttl2.set_alpha(t)

    if heading2 is not None:
        h1 = plt.suptitle(heading1, x=0.5, y=0.94)
        h2 = plt.text(*h1.get_position(), heading2)
        h2.update_from(h1)

        foot = plt.text(0.5, 0.08, 'DVE enables the use of higher dimensional unsupervised embeddings!')
        foot.update_from(h1)

        h1.set_alpha(1 - t)
        h2.set_alpha(t)


def get_match_grid(src, dest, stride):
    scatter_xy = []
    for i in i_idxs:
        for j in j_idxs:
            jj, ii = find_descriptor(j, i, src, dest, stride)
            scatter_xy.append([jj, ii])
    scatter_xy = np.array(scatter_xy)
    return scatter_xy


n_model_variations = len(model_files_all) // 2
frame = 0

imranges = [range(i, i+9) for i in range(0,n_images_to_load,9)]
for imrangei, imrange in enumerate(tqdm.tqdm(imranges)):
    for mi in tqdm.tqdm(range(n_model_variations)):
        model1 = model_files_nodve[mi]
        model2 = model_files_dve[mi]

        dest1 = [sample_descs[model1][si] for si in imrange]
        dest1_im = [sample_ims[model1][si] for si in imrange]

        dest2 = [sample_descs[model2][si] for si in imrange]
        dest2_im = [sample_ims[model2][si] for si in imrange]

        scatter_xy_1 = [get_match_grid(descs[model1], dest1i, stride) for dest1i in dest1]
        scatter_xy_2 = [get_match_grid(descs[model2], dest2i, stride) for dest2i in dest2]

        last_model_variation = mi == (n_model_variations - 1)
        new_im = mi == 0

        title1 = 'Without DVE - unstable at higher dims'
        title2 = 'With DVE - maintains robustness!'

        dimstring = ('%2d' % dest2[0].shape[0]).replace(' ', '\u00a0')
        heading = 'Matching Unsupervised Dense Embeddings with $\mathtt{%s}$ Dimensions' % dimstring

        if mi > 0 or imrangei > 0:
            for t in np.linspace(0, 1, 24):
                plt.sca(nodve_ax)
                tween_scatter(t, prev_dest1_im, dest1_im, prev_scatter_xy_1, scatter_xy_1, prev_title1, title1,
                              fade_ims=new_im, heading1=heading_prev, heading2=heading,
                              frame=frame, is_dve=False)
                plt.sca(dve_ax)
                tween_scatter(t, prev_dest2_im, dest2_im, prev_scatter_xy_2, scatter_xy_2, prev_title2, title2,
                              fade_ims=new_im, frame=frame, is_dve=True)
                plt.savefig(str(Path(args.frame_dir) / f"vis{frame:05d}.png"))
                # plt.savefig('/tmp/vis%05d.png' % frame)
                frame += 1

        plt.sca(nodve_ax)
        tween_scatter(1, dest1_im, dest1_im, scatter_xy_1, scatter_xy_1, title1, title1,
                      fade_ims=new_im, heading1=heading, heading2=heading, frame=frame, is_dve=False)
        plt.sca(dve_ax)
        tween_scatter(1, dest2_im, dest2_im, scatter_xy_2, scatter_xy_2, title2, title2,
                      fade_ims=new_im, frame=frame, is_dve=True)

        delay_len = 24 if (new_im or last_model_variation) else 1

        for delay in range(delay_len):
            plt.savefig(str(Path(args.frame_dir) / f"vis{frame:05d}.png"))
            frame += 1

        prev_dest1_im = dest1_im
        prev_dest2_im = dest2_im

        prev_scatter_xy_1 = scatter_xy_1
        prev_scatter_xy_2 = scatter_xy_2

        prev_title1 = title1
        prev_title2 = title2

        heading_prev = heading

        print(f"Processing frame: {frame}")
duration = time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - tic))
print(f"Full visualisation generation took {duration}")

# ffmpeg -i 'vis%05d.png' -pix_fmt yuv420p out.mp4 -y
