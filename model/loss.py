import torch.nn.functional as F
import time
import torch
from utils import tps



def regression_loss(prediction_normalized, meta, alpha=1., **kwargs):
    pred = prediction_normalized[0]
    kp = meta['keypts_normalized'].to(pred.device)
    B, nA, _ = pred.shape
    return F.smooth_l1_loss(pred * alpha, kp * alpha)


def segmentation_loss(x, meta, weight=None, size_average=True, **kwargs):
    target = meta["lbls"].to(x.device).long()
    n, c, h, w = x.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:  # upsample labels
        x = F.interpolate(x, size=(ht, wt), mode="bilinear", align_corners=True)
    x = x.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    return F.cross_entropy(x, target, weight=weight, reduction="mean")
    # return F.cross_entropy(x, target, weight=weight, size_average=size_average)


def dense_correlation_loss(feats, meta, pow=0.5, fold_corr=False, normalize_vectors=True):
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
        from model.folded_correlation import DenseCorr
        """This function computes the gradient explicitly to avoid the memory
        issues with using autorgrad in a for loop."""
        assert not normalize_vectors
        dense_corr = DenseCorr.apply
        return dense_corr(feats1, feats2, xxyy, batch_grid_u, stride, pow)

    loss = 0.
    for b in range(B):
        f1 = feats1[b].reshape(C, H * W)  # source
        f2 = feats2[b].reshape(C, h * w)  # target

        if normalize_vectors:
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


def dense_correlation_loss_dve(feats, meta, pow=0.5, fold_corr=False, normalize_vectors=True):
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

    if False:
        import matplotlib.pyplot as plt

        vis1 = meta['im1'][0].clone()
        vis2 = meta['im2'][0].clone()
        visgrid = tps.grid_unnormalize(grid, H_input, W_input)[0]

        fig = plt.figure()  # a new figure window
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        ax1.imshow(vis1.permute(1,2,0)+0.5)
        ax2.imshow(vis2.permute(1,2,0)+0.5)

        for i in range(H_input):
            for j in range(W_input):
                if torch.rand([]) < 0.01:
                    ax1.scatter(j,i)
                    jj,ii = visgrid[i,j]
                    ax2.scatter(jj,ii)

        dists = (batch_grid_u[0] - xxyy[::stride,::stride]).pow(2).sum(2).sqrt()
        ax3.imshow(dists/dists.max())
        fig.savefig('/tmp/lossvis.pdf')
        fig.clf()

    if fold_corr:
        """This function computes the gradient explicitly to avoid the memory
        issues with using autorgrad in a for loop."""
        from model.folded_correlation_dve import DenseCorrDve
        dense_corr = DenseCorrDve.apply
        return dense_corr(feats1, feats2, xxyy, batch_grid_u, stride,
                          normalize_vectors, pow)

    loss = 0.
    for b in range(B):
        f1 = feats1[b].reshape(C, H * W)  # source
        f2 = feats2[b].reshape(C, h * w)  # target
        fa = feats1[(b + 1) % B].reshape(C, h * w)  # auxiliary

        if normalize_vectors:
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


def dense_correlation_loss_trick(feats, meta, pow=0.5, fold_corr=False,
        normalize_vectors=True):
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
        from model.folded_correlation import DenseCorr
        """This function computes the gradient explicitly to avoid the memory
        issues with using autorgrad in a for loop."""
        assert not normalize_vectors
        dense_corr = DenseCorr.apply
        return dense_corr(feats1, feats2, xxyy, batch_grid_u, stride, pow)

    loss = 0.
    for b in range(B):
        f1 = feats1[b].reshape(C, H * W)  # source
        f2 = feats2[b].reshape(C, h * w)  # target

        if normalize_vectors:
            f1 = F.normalize(f1, p=2, dim=0) * 20
            f2 = F.normalize(f2, p=2, dim=0) * 20

        corr = torch.matmul(f1.t(), f2)
        corr = corr.reshape(H, W, h, w)

        with torch.no_grad():
            # replace with expanded terms for efficiency
            import ipdb; ipdb.set_trace()
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


def rel_diff(x1, x2, name):
    out = torch.abs(x1 - x2).sum() / torch.abs(x2).mean()
    print("rel diff for {}: {}".format(name, out))


def dense_corr_trick_check():
    dve_dim = 4
    B, C, H, W = 4, dve_dim, 4, 4

    common = {"dtype": torch.double, "requires_grad": True}
    feats = torch.randn(B, C, H, W, **common)
    batch_grid_u = torch.randn(B, H, W, 2, dtype=torch.double,
                               requires_grad=False)

    feats = feats.cuda().float()
    batch_grid_u = batch_grid_u.cuda().float()
    out = dense_correlation_loss([feats], {"grid": batch_grid_u})
    out2 = dense_correlation_loss_trick([feats], {"grid": batch_grid_u})
    rel_diff(out, out2, "trick")


if __name__ == "__main__":
    dense_corr_trick_check()
