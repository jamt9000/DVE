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

import matplotlib

if sys.platform == 'darwin':
    matplotlib.use("macosx")
import matplotlib.pyplot as plt

config_file = 'configs/celeba/smallnet-64d-dve.json'
ckpt_path = 'old_checkpoints/smallnet_celeba_64d_evc_checkpoint-epoch108.pth'

parser = argparse.ArgumentParser(description='PyTorch Template')
parser.add_argument('--config', help="config file path", default=config_file)
parser.add_argument('--resume', help='path to ckpt for evaluation', default=ckpt_path)
parser.add_argument('--device', help='indices of GPUs to enable', default='')
config = ConfigParser(parser)

# build model architecture
model = get_instance(module_arch, 'arch', config)
model.summary()

checkpoint = torch.load(ckpt_path, map_location='cpu')
state_dict = checkpoint['state_dict']
model.load_state_dict(clean_state_dict(state_dict))

model.eval()

avface = skimage.io.imread('https://www.uni-regensburg.de/Fakultaeten/phil_Fak_II/Psychologie/Psy_II/' +
                           'beautycheck/english/durchschnittsgesichter/w(01-64)_gr.jpg')
avface = Image.fromarray(avface)

dataset = data_loaders.AFLW_MTFL('data', train=False, imwidth=70)

sample_ims = []
sample_descs = []
for samplei in range(20):
    item = dataset[samplei]
    sample_im = item['data']
    sample_desc = model.forward(sample_im.unsqueeze(0))[0][0]
    sample_ims.append(sample_im)
    sample_descs.append(sample_desc)

normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769],
                                 std=[0.2599, 0.2371, 0.2323])
augmentations = []

imsize = 70
transforms = transforms.Compose([transforms.Resize((imsize, imsize)), transforms.ToTensor(), normalize])
avface_tensor = transforms(avface)

descs = model.forward(avface_tensor.unsqueeze(0))[0][0]

imC, imH, imW = avface_tensor.shape
C, H, W = descs.shape
stride = imW / W

i_idxs = np.arange(10, 60, 5)
j_idxs = np.arange(15, 60, 5)
npts = len(i_idxs) * len(j_idxs)

left = plt.subplot(1, 2, 1)
plt.imshow(norm_range(avface_tensor).permute(1, 2, 0))
rainbow = plt.cm.Spectral(np.linspace(0, 1, npts))

plt.gca().set_prop_cycle('color', rainbow)

for i in i_idxs:
    for j in j_idxs:
        plt.scatter(j, i)
plt.xticks([], [])
plt.yticks([], [])

frame = 0
for si in range(20):
    plt.subplot(1, 2, 2)
    scatter_xy = []

    dest = sample_descs[si % len(sample_descs)]
    dest_im = sample_ims[si % len(sample_descs)]

    for i in i_idxs:
        for j in j_idxs:
            jj, ii = find_descriptor(j, i, descs, dest, stride)
            scatter_xy.append([jj, ii])
    scatter_xy = np.array(scatter_xy)

    if si != 0:
        for t in np.linspace(0, 1, 24):
            plt.subplot(1, 2, 2)
            plt.cla()
            plt.xlim(left.get_xlim())
            plt.ylim(left.get_ylim())
            plt.xticks([], [])
            plt.yticks([], [])

            prev_alpha = np.maximum(0., 1 - 2 * t)
            cur_alpha = np.maximum(0., -1 + 2 * t)
            if prev_alpha:
                plt.imshow(norm_range(prev_dest_im).permute(1, 2, 0), alpha=prev_alpha)
            if cur_alpha:
                plt.imshow(norm_range(dest_im).permute(1, 2, 0), alpha=cur_alpha)

            scatter_tween = (1 - t) * prev_scatter_xy + t * scatter_xy
            plt.scatter(scatter_tween[:, 0], scatter_tween[:, 1], c=rainbow)
            plt.savefig('/tmp/%05d.png' % frame);
            frame += 1

    plt.subplot(1, 2, 2)
    plt.cla()
    plt.xlim(left.get_xlim())
    plt.ylim(left.get_ylim())
    plt.xticks([], [])
    plt.yticks([], [])

    plt.imshow(norm_range(dest_im).permute(1, 2, 0))
    plt.scatter(scatter_xy[:, 0], scatter_xy[:, 1], c=rainbow)
    for delay in range(10):
        plt.savefig('/tmp/%05d.png' % frame);
        frame += 1

    prev_dest_im = dest_im
    prev_scatter_xy = scatter_xy

    plt.imshow(norm_range(dest_im).permute(1, 2, 0))

    print('')
