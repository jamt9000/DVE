import importlib
import math
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import numpy as np

import matplotlib.pyplot as plt

plt.switch_backend('agg')
from utils.util import label_colormap
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


def keypoints_intermediate(writer, data, output, meta):
    img = norm_range(data)[0].permute(1, 2, 0)
    imH, imW, imC = img.shape

    preds, intermediates = output

    pred = preds[0].detach().cpu().clone()
    inter = intermediates[0].detach().cpu().clone()

    gt = meta['keypts'][0]

    pred[..., 0] = (pred[..., 0] + 1.) / 2. * (imW - 1)
    pred[..., 1] = (pred[..., 1] + 1.) / 2. * (imH - 1)

    inter[..., 0] = (inter[..., 0] + 1.) / 2. * (imW - 1)
    inter[..., 1] = (inter[..., 1] + 1.) / 2. * (imH - 1)

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.scatter(pred[:, 0], pred[:, 1], c='b')
    ax.scatter(gt[:, 0], gt[:, 1], c='g')
    writer.add_figure('keypoints', fig)

    fig, ax = plt.subplots()
    ax.imshow(img)
    for i in range(inter.shape[0]):
        ax.scatter(inter[i, :, 0], inter[i, :, 1])
    writer.add_figure('keypoints_intermediate', fig)


def seg_masks(writer, data, output, meta):
    N, C, H, W = output.shape
    preds = norm_range(output.max(1)[1].float()).view(N, 1, H, W)

    # res = label_colormap(preds)
    # res = make_grid(res).cpu().permute(1, 2, 0).numpy()

    res = make_grid_matshow(preds)[0].cpu().numpy()
    # pred_grid = make_grid(preds).cpu().permute(1, 2, 0).numpy()
    # from pathlib import Path ; import sys
    # sys.path.insert(0, str(Path.home() / "coding/src/zsvision/python"))
    # from zsvision.zs_iterm import zs_dispFig # NOQA
    plt.close("all")
    fig, ax = plt.subplots()
    ax.matshow(res)
    # ax.imshow(res)
    plt.axis("off")
    # for i in range(inter.shape[0]):
    #     ax.scatter(inter[i, :, 0], inter[i, :, 1])
    writer.add_figure('pred_masks', fig)


def gt_masks(writer, data, output, meta):
    N, C, H, W = data.shape
    # preds = norm_range(output.max(1)[1].float()).view(N, 1, H, W)
    gt = meta["lbls"].unsqueeze(1).to(output.device).float()
    gt = F.interpolate(gt, size=(H, W), mode="nearest")
    gt = label_colormap(gt)
    gt = make_grid(gt).cpu().permute(1, 2, 0).numpy()
    # gt = make_grid_matshow(gt)[0].cpu().numpy()
    ims = make_grid(norm_range(data)).cpu().permute(1, 2, 0).numpy()
    plt.close("all")
    fig, ax = plt.subplots()
    # ax.matshow(gt)
    ax.imshow(gt)
    ax.imshow(ims, alpha=0.5)
    plt.axis("off")
    writer.add_figure('gt_masks', fig)


def sphere_rand_proj_colormap(writer, data, output, meta):
    """Use a random projection to visualize high-dimensional embeddings
    in RGB space.
    """
    N, C, H, W = output[0].shape
    outs = output[0].clone().cpu()
    # move channels to last dimension for BMM
    outs = outs.permute((0, 2, 3, 1))
    outs = outs.reshape(N, -1, C)
    proj = torch.randn([1, C, 3]).repeat(N, 1, 1)
    projected = torch.bmm(outs, proj)
    out = projected.reshape(N, H, W, 3).permute((0, 3, 1, 2))
    normed = torch.sqrt(out[:, 0]**2. + out[:, 1]**2. + out[:, 2]**2.)[:, None]
    vis = out / normed / 2 + 0.5
    grid = make_grid(vis, nrow=8)
    writer.add_image('rand_proj_colormap/0', vis[0])
    writer.add_image('rand_proj_colormap/1', vis[1])
    writer.add_image('rand_proj_colormap', grid)


def sphere_colormap(writer, data, output, meta):
    out = output[0].cpu()[:, 0:3]
    normed = torch.sqrt(out[:, 0]**2. + out[:, 1]**2. + out[:, 2]**2.)[:, None]
    vis = out / normed / 2 + 0.5
    grid = make_grid(vis, nrow=8)
    writer.add_image('colormap/0', vis[0])
    writer.add_image('colormap/1', vis[1])
    writer.add_image('colormap', grid)

    normall = (output[0].cpu().detach()**2).sum(1).sqrt()

    fig, ax = plt.subplots()
    ms = ax.matshow(normall[0])
    fig.colorbar(ms, ax=ax)
    writer.add_figure('magnitude/0', fig)


def sphere_norm_scatter3d(writer, data, output, meta):
    output = [F.normalize(o, p=2, dim=1) for o in output]
    sphere_scatter3d(writer, data, output, meta, 'spherenorm')


def sphere_scatter3d(writer, data, output, meta, tag='sphere'):
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

    axmin = np.round(out.min())
    axmax = np.round(out.max())

    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)
    ax.set_xlim(axmin, axmax)
    ax.set_ylim(axmin, axmax)
    ax.set_zlim(axmin, axmax)
    ax.scatter3D(x0, y0, z0, c=c0.numpy(), s=40, linewidths=0, depthshade=False)
    writer.add_figure(tag + '/0', fig)

    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)
    ax.set_xlim(axmin, axmax)
    ax.set_ylim(axmin, axmax)
    ax.set_zlim(axmin, axmax)
    ax.scatter3D(x1, y1, z1, c=c1.numpy(), s=40, linewidths=0, depthshade=False)
    writer.add_figure(tag + '/1', fig)


class WriterTensorboardX():
    def __init__(self, writer_dir, logger, enable):
        self.writer = None
        if enable:
            log_path = writer_dir
            try:
                mod = importlib.import_module('tensorboardX')
                self.writer = mod.SummaryWriter(log_path)
            except ModuleNotFoundError:
                message = ("Warning: TensorboardX visualization is configured to use,"
                           "but currently not installed on this machine. "
                           "Please install the package by 'pip install tensorboardx'"
                           "command or turn off the option in the 'config.json' file.")
                logger.warning(message)
        self.step = 0
        self.mode = ''

        self.tensorboard_writer_ftns = [
            'add_scalar',
            'add_scalars',
            'add_image',
            'add_audio',
            'add_text',
            'add_histogram',
            'add_pr_curve',
            'add_embedding',
            'add_figure'
        ]

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tboard with extra info (step, tag) added.
        Otherwise:
            return blank function handle that does nothing
        """
        if name in self.tensorboard_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    add_data(
                        '{}/{}'.format(self.mode,
                                       tag),
                        data,
                        self.step,
                        *args,
                        **kwargs
                    )

            return wrapper
        else:
            # default action for returning methods defined in this class,
            # set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError(
                    "type object 'WriterTensorboardX' has no attribute '{}'".
                    format(name)
                )
            return attr


irange = range


def make_grid_matshow(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images. Modified to allow usage with matshow

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    # if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
    #     tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new_full((1, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid
