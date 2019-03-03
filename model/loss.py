import torch.nn.functional as F
import torch
from utils import tps


def dense_correlation_loss(feats, meta, device, pow=0.5):
    feats = feats[0]

    grid = meta['grid'].to(device)  # Grid (B,H,W,2): For each pixel in im1, where did it come from in im2
    flow = meta['flow'].to(device)

    H_input = grid.shape[1]
    W_input = grid.shape[2]

    feats1 = feats[0::2]
    feats2 = feats[1::2]

    B, C, H, W = feats1.shape
    h,w = H,W

    stride = H_input // H

    xxyy = tps.spatial_grid_unnormalized(H_input, W_input).to(device)

    loss = 0.

    for b in range(B):
        f1 = feats1[b].reshape(C, H * W) # source
        f2 = feats2[b].reshape(C, h * w) # target

        corr = torch.matmul(f1.t(), f2)
        corr = corr.reshape(H,W,h,w)

        grid_u = tps.grid_unnormalize(grid[b], H_input, W_input)
        diff = grid_u[:, :, None, None, :] - xxyy[None, None, :, :, :]

        # Equivalent to this
        #
        # diff = torch.zeros(H_input, W_input, H_input, W_input, 2)
        # for I in range(H_input):
        #     for J in range(W_input):
        #         for i in range(H_input):
        #             for j in range(W_input):
        #                 diff[I, J, i, j, 0] = J + flow[b, I, J, 0] - j
        #                 diff[I, J, i, j, 1] = I + flow[b, I, J, 1] - i

        diff = diff[::stride, ::stride, ::stride, ::stride]
        diff = (diff * diff).sum(4).sqrt()
        diff = diff.pow(pow)

        smcorr = F.softmax(corr.reshape(H,W,-1), dim=2).reshape(corr.shape)

        L = diff * smcorr

        loss += L.sum()

    return loss / (H*W*B)

def dense_correlation_loss_evc(feats, meta, device, pow=0.5):
    feats = feats[0]

    grid = meta['grid'].to(device)  # Grid (B,H,W,2): For each pixel in im1, where did it come from in im2
    flow = meta['flow'].to(device)

    H_input = grid.shape[1]
    W_input = grid.shape[2]

    feats1 = feats[0::2]
    feats2 = feats[1::2]

    B, C, H, W = feats1.shape
    h,w = H,W

    stride = H_input // H

    xxyy = tps.spatial_grid_unnormalized(H_input, W_input).to(device)

    loss = 0.

    for b in range(B):
        f1 = feats1[b].reshape(C, H * W).half() # source
        f2 = feats2[b].reshape(C, h * w).half() # target
        fa = feats1[(b+1) % B].reshape(C, h * w).half() # auxiliary

        corr = torch.matmul(f1.t(), fa)
        corr = corr.reshape(H,W,h,w)
        smcorr = F.softmax(corr.reshape(H,W,-1), dim=2).reshape(corr.shape)
        smcorr_fa = smcorr[...,None] * fa.reshape(H,W,1,1,-1)
        del smcorr

        f1_via_fa = smcorr_fa.sum((2,3)).reshape(C, H * W)
        del smcorr_fa

        corr2 = torch.matmul(f1_via_fa.t(), f2).reshape(corr.shape)
        smcorr2 = F.softmax(corr2.reshape(H,W,-1), dim=2).reshape(corr.shape)
        del corr2

        with torch.no_grad():
            grid_u = tps.grid_unnormalize(grid[b], H_input, W_input).half()
            diff = grid_u[:, :, None, None, :] - xxyy[None, None, :, :, :].half()

            diff = diff[::stride, ::stride, ::stride, ::stride]
            diff = (diff * diff).sum(4).sqrt()
            diff = diff.pow(pow)

        L = diff * smcorr2

        loss += L.float().sum()

    return loss / (H*W*B)