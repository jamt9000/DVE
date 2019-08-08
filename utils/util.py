import os
import torch
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # NOQA

sys.path.insert(0, str(Path.home() / "coding/src/zsvision/python"))
try:
    from zsvision.zs_iterm import zs_dispFig # NOQA
except:
    print('No zs_dispFig, figures will not be displayed in iterm')


def label_colormap(x):
    colors = np.array([
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
    ])
    ndim = len(x.shape)
    num_classes = 11
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    r = x.clone().float()
    g = x.clone().float()
    b = x.clone().float()
    if ndim == 2:
        rgb = torch.zeros((x.shape[0], x.shape[1], 3))
    else:
        rgb = torch.zeros((x.shape[0], 3, x.shape[2], x.shape[3]))
    colors = torch.from_numpy(colors)
    label_colours = dict(zip(range(num_classes), colors))

    for l in range(0, num_classes):
        r[x == l] = label_colours[l][0]
        g[x == l] = label_colours[l][1]
        b[x == l] = label_colours[l][2]
    if ndim == 2:
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
    elif ndim == 4:
        rgb[:, 0, None] = r / 255.0
        rgb[:, 1, None] = g / 255.0
        rgb[:, 2, None] = b / 255.0
    else:
        import ipdb; ipdb.set_trace()
    return rgb


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict


def get_instance(module, name, config, *args, **kwargs):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'],
                                                 **kwargs)


def coll(batch):
    b = torch.utils.data.dataloader.default_collate(batch)
    # Flatten to be 4D
    return [
        bi.reshape((-1, ) + bi.shape[-3:]) if isinstance(bi, torch.Tensor) else bi
        for bi in b
    ]


def dict_coll(batch):
    cb = torch.utils.data.dataloader.default_collate(batch)
    cb["data"] = cb["data"].reshape((-1,) + cb["data"].shape[-3:])  # Flatten to be 4D
    if False:
        from torchvision.utils import make_grid
        from utils.visualization import norm_range
        ims = norm_range(make_grid(cb["data"])).permute(1, 2, 0).cpu().numpy()
        plt.imshow(ims)
    return cb


# def dict_coll(batch):
#     b = torch.utils.data.dataloader.default_collate(batch)
#     # Flatten to be 4D
#     return [
#         bi.reshape((-1, ) + bi.shape[-3:]) if isinstance(bi, torch.Tensor) else bi
#         for bi in b
#     ]


class NoGradWrapper(nn.Module):
    def __init__(self, wrapped):
        super(NoGradWrapper, self).__init__()
        self.wrapped_module = wrapped

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.wrapped_module.forward(*args, **kwargs)


class Up(nn.Module):
    def forward(self, x):
        with torch.no_grad():
            return [F.interpolate(x[0], scale_factor=2, mode='bilinear', align_corners=False)]


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
