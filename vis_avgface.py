import sys
import skimage.io
from PIL import Image
import torch
import argparse
import model.model as module_arch
from parse_config import ConfigParser
from utils import tps, clean_state_dict, get_instance
from torchvision import transforms
from test_matching import find_descriptor
from data_loader import data_loaders
from utils.visualization import norm_range
import numpy as np
from utils.util import read_json
import os
import matplotlib
from pathlib import Path
from collections import defaultdict

if sys.platform == 'darwin':
    matplotlib.use("macosx")
import matplotlib.pyplot as plt

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

model_files_all = model_files_nodve + model_files_dve


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


models_dict = dict([(c, load_model_for_eval(c)) for c in model_files_all])

avface = skimage.io.imread('https://www.uni-regensburg.de/Fakultaeten/phil_Fak_II/Psychologie/Psy_II/' +
                           'beautycheck/english/durchschnittsgesichter/w(01-64)_gr.jpg')
avface = Image.fromarray(avface)

imsize = 70
dataset = data_loaders.AFLW_MTFL('data', train=False, imwidth=imsize)

sample_ims = defaultdict(list)
sample_descs = defaultdict(list)
for samplei in range(20):
    for m in model_files_all:
        model = models_dict[m]
        item = dataset[samplei]
        sample_im = item['data']
        sample_desc = model.forward(sample_im.unsqueeze(0))[0][0]

        sample_ims[m].append(sample_im)
        sample_descs[m].append(sample_desc)

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

plt.figure(figsize=(9,3))

left = plt.subplot(1, 3, 1)
plt.imshow(norm_range(avface_tensor).permute(1, 2, 0))
rainbow = plt.cm.Spectral(np.linspace(0, 1, npts))
plt.title('Query')
plt.gca().set_prop_cycle('color', rainbow)

for i in i_idxs:
    for j in j_idxs:
        plt.scatter(j, i)
plt.xticks([], [])
plt.yticks([], [])


def ax_reset():
    plt.cla()
    plt.xlim(left.get_xlim())
    plt.ylim(left.get_ylim())
    plt.xticks([], [])
    plt.yticks([], [])


def tween_scatter(t, im1, im2, scatter1, scatter2, title1, title2, fade_ims=True):
    ax_reset()

    if fade_ims:
        prev_alpha = np.maximum(0., 1 - 2 * t)
        cur_alpha = np.maximum(0., -1 + 2 * t)
    else:
        prev_alpha = 0.
        cur_alpha = 1.

    if prev_alpha:
        plt.imshow(norm_range(im1).permute(1, 2, 0), alpha=prev_alpha)
    if cur_alpha:
        plt.imshow(norm_range(im2).permute(1, 2, 0), alpha=cur_alpha)

    ease = (-np.cos(np.pi*t) + 1)/2
    scatter_tween = (1 - ease) * scatter1 + ease * scatter2
    plt.scatter(scatter_tween[:, 0], scatter_tween[:, 1], c=rainbow)
    ttl1 = plt.title(title1, loc='left')
    ttl2 = plt.text(*ttl1.get_position(), title2)
    ttl2.update_from(ttl1)
    ttl1.set_alpha(1 - t)
    ttl2.set_alpha(t)


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
for si in range(20):
    for mi in range(n_model_variations):
        model1 = model_files_nodve[mi]
        model2 = model_files_dve[mi]

        plt.subplot(1, 3, 2)

        dest1 = sample_descs[model1][si]
        dest1_im = sample_ims[model1][si]

        dest2 = sample_descs[model2][si]
        dest2_im = sample_ims[model2][si]

        scatter_xy_1 = get_match_grid(descs[model1], dest1, stride)
        scatter_xy_2 = get_match_grid(descs[model2], dest2, stride)

        last_model_variation = mi == (n_model_variations - 1)
        new_im = mi == 0

        title1 = ('              %dD' % dest1.shape[0]).replace('3D', '  3D')
        title2 = ('         %dD + DVE' % dest2.shape[0]).replace('3D', '  3D')

        if mi > 0 or si > 0:
            for t in np.linspace(0, 1, 24):
                plt.subplot(1, 3, 2)
                tween_scatter(t, prev_dest1_im, dest1_im, prev_scatter_xy_1, scatter_xy_1, prev_title1, title1,
                              fade_ims=new_im)
                plt.subplot(1, 3, 3)
                tween_scatter(t, prev_dest2_im, dest2_im, prev_scatter_xy_2, scatter_xy_2, prev_title2, title2,
                              fade_ims=new_im)
                plt.savefig('/tmp/vis%05d.png' % frame)
                frame += 1

        plt.subplot(1, 3, 2)
        ax_reset()
        plt.imshow(norm_range(dest1_im).permute(1, 2, 0))
        plt.scatter(scatter_xy_1[:, 0], scatter_xy_1[:, 1], c=rainbow)
        plt.title(title1, loc='left')

        plt.subplot(1, 3, 3)
        ax_reset()
        plt.imshow(norm_range(dest2_im).permute(1, 2, 0))
        plt.scatter(scatter_xy_2[:, 0], scatter_xy_2[:, 1], c=rainbow)
        plt.title(title2, loc='left')

        delay_len = 24 if (new_im or last_model_variation) else 1

        for delay in range(delay_len):
            plt.savefig('/tmp/vis%05d.png' % frame);
            frame += 1

        prev_dest1_im = dest1_im
        prev_dest2_im = dest2_im

        prev_scatter_xy_1 = scatter_xy_1
        prev_scatter_xy_2 = scatter_xy_2

        prev_title1 = title1
        prev_title2 = title2

        print(frame)

# ffmpeg -i 'vis%05d.png' -pix_fmt yuv420p out.mp4 -y
