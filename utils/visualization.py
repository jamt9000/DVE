import importlib
import torch
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

plt.switch_backend('agg')
from mpl_toolkits import mplot3d


def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min + 1e-5)


def norm_range(t, range=None):
    t = t.clone()
    if range is not None:
        norm_ip(t, range[0], range[1])
    else:
        norm_ip(t, float(t.min()), float(t.max()))
    return t


def sphere_colormap(writer, data, output):
    out = output[0].cpu()[:, 0:3]
    normed = torch.sqrt(out[:, 0] ** 2. + out[:, 1] ** 2. + out[:, 2] ** 2.)[:, None]
    vis = out / normed / 2 + 0.5
    grid = make_grid(vis, nrow=8)
    writer.add_image('colormap/0', vis[0])
    writer.add_image('colormap/1', vis[1])
    writer.add_image('colormap', grid)


def sphere_scatter3d(writer, data, output):
    data = norm_range(data)
    out = output[0].cpu().detach().clone()
    out0 = out[0][0:3]
    out1 = out[1][0:3]

    stride = data.shape[2] // out.shape[2]

    im0 = data[0][:, ::stride, ::stride]
    x0 = out0[0].reshape(-1)
    y0 = out0[1].reshape(-1)
    z0 = out0[2].reshape(-1)
    c0 = im0.permute(1, 2, 0).reshape(-1, 3)

    im1 = data[1][:, ::stride, ::stride]
    x1 = out1[0].reshape(-1)
    y1 = out1[1].reshape(-1)
    z1 = out1[2].reshape(-1)
    c1 = im1.permute(1, 2, 0).reshape(-1, 3)

    axmin = out.min()
    axmax = out.max()

    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)
    ax.set_xlim(axmin, axmax)
    ax.set_ylim(axmin, axmax)
    ax.set_zlim(axmin, axmax)
    ax.scatter3D(x0, y0, z0, c=c0.numpy(), s=40, linewidths=0,
                 depthshade=False)
    writer.add_figure('sphere/0', fig)

    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)
    ax.set_xlim(axmin, axmax)
    ax.set_ylim(axmin, axmax)
    ax.set_zlim(axmin, axmax)
    ax.scatter3D(x1, y1, z1, c=c1.numpy(), s=40, linewidths=0,
                 depthshade=False)
    writer.add_figure('sphere/1', fig)


class WriterTensorboardX():
    def __init__(self, writer_dir, logger, enable):
        self.writer = None
        if enable:
            log_path = writer_dir
            try:
                self.writer = importlib.import_module('tensorboardX').SummaryWriter(log_path)
            except ModuleNotFoundError:
                message = "Warning: TensorboardX visualization is configured to use, but currently not installed on this machine. " + \
                          "Please install the package by 'pip install tensorboardx' command or turn off the option in the 'config.json' file."
                logger.warning(message)
        self.step = 0
        self.mode = ''

        self.tensorboard_writer_ftns = ['add_scalar', 'add_scalars', 'add_image', 'add_audio', 'add_text',
                                        'add_histogram', 'add_pr_curve', 'add_embedding', 'add_figure']

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return blank function handle that does nothing
        """
        if name in self.tensorboard_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    add_data('{}/{}'.format(self.mode, tag), data, self.step, *args, **kwargs)

            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object 'WriterTensorboardX' has no attribute '{}'".format(name))
            return attr
