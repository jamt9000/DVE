import torch.nn.functional as F
import time
import torch
import torch.nn.functional as F
from utils import tps

from model.folded_correlation import DenseCorr
from model.folded_correlation_evc import DenseCorrEvc

def regression_loss(prediction_normalized, meta, **kwargs):
    pred = prediction_normalized[0]
    kp = meta['keypts_normalized'].to(pred.device)
    B, nA, _ = pred.shape
    return F.smooth_l1_loss(pred, kp)


def dense_correlation_loss(feats, meta, pow=0.5, fold_corr=False):
    feats = feats[0]
    device = feats.device
    grid = meta['grid']

    # Grid (B,H,W,2): For each pixel in im1, where did it come from in im2
    grid = grid.to(device)

    H_input = grid.shape[1]
    W_input = grid.shape[2]

    feats1 = feats[0::2]
    feats2 = feats[1::2]

    B, C, H, W = feats1.shape
    h, w = H, W

    stride = H_input // H

    batch_grid_u = tps.grid_unnormalize(grid, H_input, W_input)
    batch_grid_u = batch_grid_u[:, ::stride, ::stride, :]
    xxyy = tps.spatial_grid_unnormalized(H_input, W_input).to(device)

    if fold_corr:
        """This function computes the gradient explicitly to avoid the memory
        issues with using autorgrad in a for loop."""
        dense_corr = DenseCorr.apply
        return dense_corr(feats1, feats2, xxyy, batch_grid_u, stride, pow)

    loss = 0.
    for b in range(B):
        f1 = feats1[b].reshape(C, H * W)  # source
        f2 = feats2[b].reshape(C, h * w)  # target

        f1 = F.normalize(f1, p=2, dim=0) * 20
        f2 = F.normalize(f2, p=2, dim=0) * 20

        corr = torch.matmul(f1.t(), f2)
        corr = corr.reshape(H, W, h, w)

        with torch.no_grad():
            diff = batch_grid_u[b, :, :, None, None, :] - \
                    xxyy[None, None, ::stride, ::stride, :]
            diff = (diff * diff).sum(4).sqrt()
            diff = diff.pow(pow)

        # grid_u = tps.grid_unnormalize(grid[b], H_input, W_input)
        # diff = grid_u[:, :, None, None, :] - xxyy[None, None, :, :, :]

        # Equivalent to this
        #
        # diff = torch.zeros(H_input, W_input, H_input, W_input, 2)
        # for I in range(H_input):
        #     for J in range(W_input):
        #         for i in range(H_input):
        #             for j in range(W_input):
        #                 diff[I, J, i, j, 0] = J + flow[b, I, J, 0] - j
        #                 diff[I, J, i, j, 1] = I + flow[b, I, J, 1] - i

        # diff = diff[::stride, ::stride, ::stride, ::stride]
        # diff = (diff * diff).sum(4).sqrt()
        # diff = diff.pow(pow)

        smcorr = F.softmax(corr.reshape(H, W, -1), dim=2).reshape(corr.shape)

        L = diff * smcorr

        loss += L.sum()

    return loss / (H * W * B)


def estimate_mem(x):
    if x.dtype == torch.float32:
        nbytes = 4
    elif x.dtype == torch.float16:
        nbytes = 2
    elif x.dtype == torch.int8:
        nbytes = 1
    else:
        import ipdb; ipdb.set_trace()
    return torch.numel(x) * nbytes / (1024) ** 3


def dense_correlation_loss_evc(feats, meta, pow=0.5, fold_corr=False):
    feats = feats[0]
    device = feats.device

    # Grid (B,H,W,2): For each pixel in im1, where did it come from in im2
    grid = meta['grid'].to(device)

    H_input = grid.shape[1]
    W_input = grid.shape[2]

    feats1 = feats[0::2]
    feats2 = feats[1::2]

    B, C, H, W = feats1.shape
    h, w = H, W

    stride = H_input // H

    xxyy = tps.spatial_grid_unnormalized(H_input, W_input).to(device)
    batch_grid_u = tps.grid_unnormalize(grid, H_input, W_input)
    batch_grid_u = batch_grid_u[:, ::stride, ::stride, :]

    if fold_corr:
        """This function computes the gradient explicitly to avoid the memory
        issues with using autorgrad in a for loop."""
        dense_corr = DenseCorrEvc.apply
        return dense_corr(feats1, feats2, xxyy, batch_grid_u, stride, pow)

    loss = 0.
    for b in range(B):
        f1 = feats1[b].reshape(C, H * W)  # source
        f2 = feats2[b].reshape(C, h * w)  # target
        fa = feats1[(b + 1) % B].reshape(C, h * w)  # auxiliary

        f1 = F.normalize(f1, p=2, dim=0) * 20
        f2 = F.normalize(f2, p=2, dim=0) * 20
        fa = F.normalize(fa, p=2, dim=0) * 20

        corr = torch.matmul(f1.t(), fa)
        corr = corr.reshape(H, W, h, w)
        smcorr = F.softmax(corr.reshape(H, W, -1), dim=2).reshape(corr.shape)
        smcorr_fa = smcorr[None, ...] * fa.reshape(-1, 1, 1, h, w)
        del smcorr

        f1_via_fa = smcorr_fa.sum((3, 4)).reshape(C, H * W)
        del smcorr_fa

        corr2 = torch.matmul(f1_via_fa.t(), f2).reshape(corr.shape)
        smcorr2 = F.softmax(corr2.reshape(H, W, -1), dim=2).reshape(corr.shape)
        del corr2

        with torch.no_grad():
            diff = batch_grid_u[b, :, :, None, None, :] - \
                    xxyy[None, None, ::stride, ::stride, :]
            diff = (diff * diff).sum(4).sqrt()
            diff = diff.pow(pow)

        L = diff * smcorr2

        loss += L.float().sum()

    return loss / (H * W * B)
