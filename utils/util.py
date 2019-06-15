import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


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
            return [F.interpolate(x[0], scale_factor=2, mode='bilinear')]
