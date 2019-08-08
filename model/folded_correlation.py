import torch
import argparse
import torch.nn.functional as F
import copy
from utils import tps
import time
from collections import defaultdict
from torch.autograd import gradcheck

LOCAL_CHECKS = False
PROFILE = False


class DenseCorr(torch.autograd.Function):

    @staticmethod
    def forward(ctx, feats1, feats2, xxyy, batch_grid_u, stride, pow=0.5):
        """Compute the folded dense correlation loss forward pass.

        Args:
            feats1 (torch.Tensor): N x C x h x h tensor of features
            feats2 (torch.Tensor): N x C x h x w tensor of features
            xxyy (torch.Tensor): H x W x 2 grid of uniform sampling locations.
            batch_grid_u (torch.Tensor): N x h x w x 2 grid of sampling
                locations.
            stride (int): the stride to be applied to the image grid to match
                the spatial dimensions of the features (so that
                `H = h * stride`).
            pow (float :: 0.5): power by which to raise the root distances
                between pixel locations.

        Returns:
            (torch.Tensor): The total loss for the given minibatch of inputs.
        """
        with torch.no_grad():
            B, C, H, W = feats1.shape
            params = torch.IntTensor([B, C, H, W, stride])
            pow_tensor = torch.FloatTensor([pow])
            ctx.save_for_backward(feats1, feats2, xxyy, batch_grid_u,
                                  params, pow_tensor)

            loss = 0.
            for b in range(B):
                f1 = feats1[b].reshape(C, H * W)  # source
                f2 = feats2[b].reshape(C, H * W)  # target
                corr = torch.matmul(f1.t(), f2)
                corr = corr.reshape(H, W, H, W)
                diff = batch_grid_u[b, :, :, None, None, :] - \
                    xxyy[None, None, ::stride, ::stride, :]
                diff = (diff * diff).sum(4).sqrt()
                diff = diff.pow(pow)
                smcorr = F.softmax(corr.reshape(H, W, -1), dim=2)
                smcorr = smcorr.reshape(corr.shape)
                L = diff * smcorr
                loss += L.sum()
        return loss / (H * W * B)

    @staticmethod
    def backward(ctx, grad_output):
        """Compute the folded dense correlation loss backward pass.

        Args:
            (torch.Tensor): The gradient of the total loss with respect to the
                output of the dense correlation loss.

        Returns:
            (torch.Tensor): N x C x h x h tensor of gradients
            (torch.Tensor): N x C x h x w tensor of gradients
            (None): H x W x 2 grid of uniform sampling locations
            (None): no gradient for `xxyy`
            (None): no gradient for `batch_grid_u`
            (None): no gradient for `stride`
            (None): no gradient for `pow`
        """
        if PROFILE:
            batch_tic = time.time()
            tic = time.time()
            timings = defaultdict(float)

        feats1, feats2, xxyy, batch_grid_u, params, pow = ctx.saved_tensors

        """We needed to store the integers as part of a tensor, so the
        unpacking code here is a little convoluted."""
        B, C, H, W, stride = [x.item() for x in params]
        pow = pow.item()

        """This is a pattern that is very convenient - at the top of backward
        unpack saved_tensors and initialize all gradients w.r.t. inputs to
        None. Thanks to the fact that additional trailing Nones are
        ignored, the return statement is simple even when the function has
        optional inputs."""
        grad_feats1 = grad_feats2 = grad_xxyy = grad_batch_u = None
        grad_stride = grad_pow = None

        """Returning gradients for inputs that don't require it is
        not an error."""
        assert ctx.needs_input_grad[0], "expected feats1 to need grad"
        assert ctx.needs_input_grad[1], "expected feats2 to need grad"
        assert not ctx.needs_input_grad[2], "expected xxyy does not need grad"
        assert not ctx.needs_input_grad[3], "expected batch_grid_u does not need grad"
        assert not ctx.needs_input_grad[4], "expected stride does not need grad"

        if PROFILE:
            timings["back-init"] = time.time() - tic
            tic = time.time()

        with torch.no_grad():

            if feats1.is_cuda:
                # TODO: clean up types here
                if feats1.dtype == torch.float32:
                    grad_feats1 = torch.cuda.FloatTensor(B, C, H, W).fill_(0)
                    grad_feats2 = torch.cuda.FloatTensor(B, C, H, W).fill_(0)
                elif feats1.dtype == torch.float16:
                    grad_feats1 = torch.cuda.HalfTensor(B, C, H, W).fill_(0)
                    grad_feats2 = torch.cuda.HalfTensor(B, C, H, W).fill_(0)
            else:
                grad_feats1 = torch.zeros((B, C, H, W), dtype=feats1.dtype)
                grad_feats2 = torch.zeros((B, C, H, W), dtype=feats2.dtype)

            grad_loss = grad_output / (H * W * B)

            if PROFILE:
                timings["data transfer"] = time.time() - batch_tic

            for b in range(B):

                if PROFILE:
                    tic = time.time()

                with torch.no_grad():
                    diff = batch_grid_u[b, :, :, None, None, :] - \
                           xxyy[None, None, ::stride, ::stride, :]
                    diff = (diff * diff).sum(4).sqrt()
                    diff = diff.pow(pow)

                if PROFILE:
                    timings["diff-grid"] += time.time() - tic
                    tic = time.time()

                # loss gradient for the current minibatch element (expand to tensor)
                grad_loss_b = grad_loss
                grad_smcorr = grad_loss_b * diff

                if LOCAL_CHECKS:
                    grad_loss_b_ = torch.ones(diff.shape, dtype=diff.dtype) * grad_loss
                    smcorr_ = torch.randn(diff.shape, dtype=torch.double, requires_grad=True)
                    with torch.autograd.enable_grad():
                        L_ = diff * smcorr_
                        d_smcorr = torch.autograd.grad(
                            outputs=L_,
                            inputs=smcorr_,
                            grad_outputs=grad_loss_b_,
                        )
                        grad_smcorr = grad_loss_b * diff
                        rel_diff(grad_smcorr, d_smcorr, "smax")

                grad_smcorr = grad_smcorr.view(H, W, -1)
                f1_ = feats1[b].view(C, H * W)
                f2_ = feats2[b].view(C, H * W)

                if PROFILE:
                    timings["scale-feats"] += time.time() - tic
                    tic = time.time()

                # This is where the memory usage gets serious
                corr = torch.matmul(f1_.t(), f2_)

                if PROFILE:
                    timings["correlation"] += time.time() - tic
                    tic = time.time()

                # Direct backward pass for softmax
                corr = corr.view(H, W, -1)
                smcorr = F.softmax(corr, dim=2)
                smcorr = smcorr.view(corr.shape)
                sum_term = torch.sum(grad_smcorr * smcorr, dim=2, keepdim=True)
                grad_corr = smcorr * (grad_smcorr - sum_term)

                if PROFILE:
                    timings["softmax"] += time.time() - tic
                    tic = time.time()

                # safety checks
                if LOCAL_CHECKS:
                    with torch.enable_grad():
                        corr_num = corr.clone().requires_grad_()
                        smcorr_num = F.softmax(corr_num, dim=2)
                        grad_corr_num = torch.autograd.grad(
                            outputs=smcorr_num,
                            inputs=(corr_num,),
                            grad_outputs=grad_smcorr,
                        )
                        rel_diff(grad_corr, grad_corr_num[0], "smax")

                """The main work is done by some fairly beefy MM ops to compute
                pairwise correlations:"""
                grad_corr = grad_corr.view(H * W, H * W)
                grad_f1 = torch.matmul(grad_corr, f2_.t()).t()
                grad_f2 = torch.matmul(f1_, grad_corr)

                if PROFILE:
                    timings["corr-back"] += time.time() - tic
                    tic = time.time()

                if LOCAL_CHECKS:
                    with torch.enable_grad():
                        f1_num = f1_.clone().requires_grad_()
                        f2_num = f2_.clone().requires_grad_()
                        corr_num = torch.matmul(f1_num.t(), f2_num)
                        grad_f1_num, grad_f2_num = torch.autograd.grad(
                            outputs=corr_num,
                            inputs=(f1_num, f2_num),
                            grad_outputs=grad_corr,
                        )
                        rel_diff(grad_f1, grad_f1_num, "corr->f1")
                        rel_diff(grad_f2, grad_f2_num, "corr->f2")
                        grad_f1_inner = grad_f1_num
                        grad_f2_inner = grad_f2_num

                # safety checks over the whole inner loop
                if LOCAL_CHECKS:
                    with torch.enable_grad():

                        f1_num = feats1[b].clone().detach().requires_grad_()
                        f2_num = feats2[b].clone().detach().requires_grad_()

                        # BLock 1 ------------------------------------------
                        f1_num = f1_num.reshape(C, H * W)  # source
                        f2_num = f2_num.reshape(C, H * W)  # target
                        # BLock 1 ------------------------------------------

                        # BLock 2 ------------------------------------------
                        corr_num = torch.matmul(f1_num.t(), f2_num)
                        corr_num = corr_num.reshape(H, W, H, W)
                        # BLock 2 ------------------------------------------
                        corr_num = corr_num.reshape(H, W, -1)
                        smcorr_num = F.softmax(corr_num, dim=2)
                        smcorr_num = smcorr_num.reshape(corr_num.shape)

                        grad_f1_num, grad_f2_num = torch.autograd.grad(
                            outputs=smcorr_num,
                            inputs=(f1_num, f2_num),
                            grad_outputs=grad_smcorr,
                        )
                        grad_f1_outer = grad_f1_num
                        grad_f2_outer = grad_f2_num

                        rel_diff(grad_f1, grad_f1_num, "df1_")
                        rel_diff(grad_f2, grad_f2_num, "df2_")

                grad_feats1[b] = grad_f1.reshape((C, H, W))
                grad_feats2[b] = grad_f2.reshape((C, H, W))
                if PROFILE:
                    timings["feat-assign"] += time.time() - tic

            """Distribute the gradients back among the input tensor features that
            require them."""
            # grad_feats1 = grad_feats1.unsqueeze(0).repeat(B, 1, 1, 1)
            # grad_feats2 = grad_feats2.unsqueeze(0).repeat(B, 1, 1, 1)

            if LOCAL_CHECKS:
                with torch.enable_grad():
                    loss = 0.
                    grad_loss_ = grad_loss * (H * W * B)  # unscale
                    for b in range(B):
                        f1 = feats1[b].reshape(C, H * W)  # source
                        f2 = feats2[b].reshape(C, H * W)  # target

                        corr = torch.matmul(f1.t(), f2)
                        corr = corr.reshape(H, W, H, W)

                        with torch.no_grad():
                            diff = batch_grid_u[b, :, :, None, None, :] - \
                                xxyy[None, None, ::stride, ::stride, :]
                            diff = (diff * diff).sum(4).sqrt()
                            diff = diff.pow(pow)

                        smcorr = F.softmax(corr.reshape(H, W, -1), dim=2).reshape(corr.shape)
                        L = diff * smcorr
                        loss += L.sum()
                    loss = loss / (H * W * B)
                    grad_f1_num, grad_f2_num = torch.autograd.grad(
                        outputs=loss,
                        inputs=(feats1, feats2),
                        grad_outputs=grad_loss_,
                    )
                    rel_diff(grad_feats1, grad_f1_num, "full-loop f2")
                    rel_diff(grad_feats2, grad_f2_num, "full-loop f2")
            if PROFILE:
                tic = time.time()

            """Clear up all intermediate structures to avoid autograd
            implosions."""
            del grad_loss_b
            del b
            del grad_f1
            del grad_f2
            del smcorr
            del corr
            del diff
            del params

            if PROFILE:
                timings["cleanup"] += time.time() - tic

            if PROFILE:
                timings["minibatch"] = time.time() - batch_tic
                print("==============")
                total_ratios = 0
                for key in timings:
                    ratio = 100 * timings[key] / timings["minibatch"]
                    msg = "{:.3f} ({:.2f}%) >>> {}"
                    print(msg.format(timings[key], ratio, key))
                    total_ratios += ratio
                msg = "{:.3f}s >>> ratio total {}"
                print(msg.format(timings["minibatch"], total_ratios - 100))
                print("==============")

        return (grad_feats1, grad_feats2, grad_xxyy, grad_batch_u,
                grad_stride, grad_pow)


def rel_diff(x1, x2, name):
    out = torch.abs(x1 - x2).sum() / torch.abs(x2).mean()
    print("rel diff for {}: {}".format(name, out))


def dense_corr_check():
    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    dense_corr = DenseCorr.apply
    dve_dim = 4
    stride = 2
    B, C, H, W = 4, dve_dim, 4, 4

    common = {"dtype": torch.double, "requires_grad": True}
    feats1 = torch.randn(B, C, H, W, **common)
    feats2 = torch.randn(B, C, H, W, **common)

    batch_grid_u = torch.randn(B, H, W, 2, dtype=torch.double,
                               requires_grad=False)

    H_input = H * stride
    W_input = W * stride
    xxyy = tps.spatial_grid_unnormalized(H_input, W_input).double()
    xxyy.requires_grad = False
    args = (feats1, feats2, xxyy, batch_grid_u, stride)

    feats1.cuda()
    feats2.cuda()
    xxyy.cuda()
    batch_grid_u.cuda()
    test = gradcheck(dense_corr, args, eps=1e-6, atol=1e-4)
    print("passed test: {}".format(test))


if __name__ == "__main__":
    dense_corr_check()
