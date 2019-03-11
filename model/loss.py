import torch.nn.functional as F
import time
import torch
from utils import tps

from dense_corr_back import DenseCorr

USE_HALF = False


def dense_correlation_loss(feats, meta, pow=0.5):
    feats = feats[0]
    device = feats.device


    grid = meta['grid']

    if USE_HALF:
        grid = grid.half()
        feats = feats.half()

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
    if USE_HALF:
        xxyy = xxyy.half()

    if True:
        dense_corr = DenseCorr.apply
        loss = dense_corr(feats1, feats2, xxyy, batch_grid_u, stride, pow)
        return loss
        # import ipdb; ipdb.set_trace()
    else:
        loss = 0.
        for b in range(B):
            f1 = feats1[b].reshape(C, H * W)  # source
            f2 = feats2[b].reshape(C, h * w)  # target

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


def dense_correlation_loss_evc(feats, meta, pow=0.5):
    feats = feats[0]
    device = feats.device

    grid = meta['grid'].to(device)  # Grid (B,H,W,2): For each pixel in im1, where did it come from in im2
    # flow = meta['flow'].to(device)

    H_input = grid.shape[1]
    W_input = grid.shape[2]

    feats1 = feats[0::2]
    feats2 = feats[1::2]
    import ipdb; ipdb.set_trace()

    B, C, H, W = feats1.shape
    h, w = H, W

    stride = H_input // H

    xxyy = tps.spatial_grid_unnormalized(H_input, W_input).to(device)

    tic = time.time()
    batch_grid_u = tps.grid_unnormalize(grid, H_input, W_input)
    batch_grid_u = batch_grid_u[:, ::stride, ::stride, :]
    # import ipdb; ipdb.set_trace()
    # diff = batch_grid_u[:, :, :, None, None, :] - xxyy[None, None, None, :, :, :]
    # diff = diff[:, ::stride, ::stride, ::stride, ::stride]
    # diff = batch_grid_u[:, ::stride, ::stride, None, None, :] \
            # - xxyy[None, None, None, ::stride, ::stride, :]
    # diff = diff[:, ::stride, ::stride, ::stride, ::stride]
    # diff = (diff * diff).sum(5).sqrt()
    # diff = diff.pow(pow)
    # timings["batch-grid"] = time.time() - tic


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

        xx = locals()
        for name, value in xx.items():
            if hasattr(value, "shape"):
                mem = estimate_mem(value)
                print("{} -> {} {:.3f}GiB".format(name, value.shape, mem))
        import ipdb; ipdb.set_trace()


    return loss / (H * W * B)
