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
from utils.util import read_json, pad_and_crop
import os
import matplotlib
from pathlib import Path
from collections import defaultdict
import shelve

# matplotlib.font_manager._rebuild()
matplotlib.rc('font', family='serif', serif='cmr10')
# I downloaded the bold version of Space Mono to get a bold & monospace at the same time in mathtext
matplotlib.rc('font', monospace='Space Mono, Andale Mono')
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
parser.add_argument("--save_hq_ims", action="store_true")
parser.add_argument("--hq_frame_snapshot", type=int, default=96)

args = parser.parse_args()

model_files_nodve = ['data/models/celeba-smallnet-64d/2019-08-04_17-56-04/checkpoint-epoch100.pth']
model_files_dve = ['data/models/celeba-smallnet-64d-dve/2019-08-02_06-20-28/checkpoint-epoch100.pth']
model_files_all = model_files_nodve + model_files_dve

def grow_axis(ax, d):
    l, b, r, t = ax.get_position().extents
    ax.set_position(matplotlib.transforms.Bbox.from_extents((l - d, b - d, r + d, t + d)))


def nudge_axis(ax, d):
    l, b, r, t = ax.get_position().extents
    ax.set_position(matplotlib.transforms.Bbox.from_extents((l + d, b, r + d, t)))


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

imsize = 70
n_images_to_load = 100
#dataset = data_loaders.MAFLAligned(root='data/celeba', train=False, imwidth=100, crop=15, use_hq_ims=False)
dataset = data_loaders.AFLW_MTFL('data', train=False, imwidth=imsize, crop=0)

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
        sample_descs[m] = sample_descs.get(m, []) + [sample_desc]

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

#i_idxs = np.arange(10, 66, 6) -1
#j_idxs = np.arange(10, 66, 6)  -1


npts = len(i_idxs) * len(j_idxs)

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


plt.sca(dve_ax)
ax_reset()
plt.gca().set_prop_cycle('color', rainbow)
plt.xlabel('DVE')
plt.sca(nodve_ax)
ax_reset()
plt.gca().set_prop_cycle('color', rainbow)
plt.xlabel('No DVE')


model1 = model_files_nodve[-1]
model2 = model_files_dve[-1]

si = 0
for i in i_idxs:
    for j in j_idxs:
        dest1 = sample_descs[model1][si]
        dest1_im = sample_ims[model1][si]
        dest1_im_numpy = norm_range(dest1_im).permute(1, 2, 0).numpy()

        dest2 = sample_descs[model2][si]
        dest2_im = sample_ims[model2][si]
        dest2_im_numpy = norm_range(dest2_im).permute(1, 2, 0).numpy()

        jj, ii = find_descriptor(j, i, descs[model1], dest1, stride)
        jj = int(jj)
        ii = int(ii)

        jj2, ii2 = find_descriptor(j, i, descs[model2], dest2, stride)
        jj2 = int(jj2)
        ii2 = int(ii2)


        ctx = 15
        sz = 2.5

        plt.sca(nodve_ax)
        imcrop1 = pad_and_crop(dest1_im_numpy, [ii - ctx, ii + ctx, jj - ctx, jj + ctx])
        plt.imshow(imcrop1, extent=[j - sz, j + sz, i + sz, i - sz])  # lrbt
        if np.sqrt((ii-ii2)**2+(jj-jj2)**2) > 8:
            plt.gca().add_patch(plt.Rectangle((j-sz,i-sz),sz*2,sz*2,linewidth=2,edgecolor='r',facecolor='none'))
        fac = plt.gca().get_position().width / nodve_ax.get_position().width
        plt.scatter(j, i, s=(matplotlib.rcParams['lines.markersize'] * fac) ** 2)

        plt.sca(dve_ax)
        imcrop2 = pad_and_crop(dest2_im_numpy, [ii2 - ctx, ii2 + ctx, jj2 - ctx, jj2 + ctx])
        plt.imshow(imcrop2, extent=[j - sz, j + sz, i + sz, i - sz])  # lrbt
        fac = plt.gca().get_position().width / nodve_ax.get_position().width
        plt.scatter(j, i, s=(matplotlib.rcParams['lines.markersize'] * fac) ** 2)



        si += 1
plt.show()
print('done')