import torch
import argparse
import torch.nn.functional as F
import copy
from utils import tps
import time
from collections import defaultdict
from torch.autograd import gradcheck

LOCAL_CHECKS = False


def rel_diff(x1, x2, name):
    out = torch.abs(x1 - x2).sum() / torch.abs(x2).mean()
    print("rel diff for {}: {}".format(name, out))


# Inherit from Function
class LinearFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, x, weight, bias=None):
        ctx.save_for_backward(x, weight, bias)
        output = x.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        x, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(x)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


class MulConstant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, constant):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.constant = constant
        return tensor * constant

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output * ctx.constant, None


def linear_check():
    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    linear = LinearFunction.apply
    x = (torch.randn(20,20,dtype=torch.double,requires_grad=True),
	 torch.randn(30,20,dtype=torch.double,requires_grad=True))
    test = gradcheck(linear, x, eps=1e-6, atol=1e-4)
    print(test)


class DenseCorr(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, feats1, feats2, xxyy, batch_grid_u, stride, pow=0.5):
        with torch.no_grad():
            B, C, H, W = feats1.shape
            params = torch.IntTensor([B, C, H, W, stride])
            pow_tensor = torch.FloatTensor([pow])
            ctx.save_for_backward(feats1, feats2, batch_grid_u, xxyy, params, pow_tensor)

            loss = 0.
            for b in range(B):
                f1 = feats1[b].reshape(C, H * W)  # source
                f2 = feats2[b].reshape(C, H * W)  # target

                corr = torch.matmul(f1.t(), f2)
                corr = corr.reshape(H, W, H, W)

                # with torch.no_grad():
                    # diff = batch_grid_u[b, :, :, None, None, :] - \
                        # xxyy[None, None, ::stride, ::stride, :]
                    # diff = (diff * diff).sum(4).sqrt()
                    # diff = diff.pow(pow)
                diff = batch_grid_u[b, :, :, None, None, :] - \
                    xxyy[None, None, ::stride, ::stride, :]
                diff = (diff * diff).sum(4).sqrt()
                diff = diff.pow(pow)

                smcorr = F.softmax(corr.reshape(H, W, -1), dim=2).reshape(corr.shape)
                L = diff * smcorr
                loss += L.sum()
        return loss / (H * W * B)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        batch_tic = time.time()
        tic = time.time()
        timings = defaultdict(float)

        feats1, feats2, batch_grid_u, xxyy, params, pow = ctx.saved_tensors

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
        assert not ctx.needs_input_grad[2], "expected xxyy not to need grad"
        assert not ctx.needs_input_grad[3], "expected batch_grid_u not to need grad"
        assert not ctx.needs_input_grad[4], "expected stride not to need grad"
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

            # tic = time.time()

            # TODO(clean up memory usage)
            # feats1_ = feats1.clone().detach()
            # feats2_ = feats2.clone().detach()
            feats1_ = feats1
            feats2_ = feats2
            timings["data transfer"] = time.time() - batch_tic

            for b in range(B):

                tic = time.time()
                # ----------------------------------------
                with torch.no_grad():
                    diff = batch_grid_u[b, :, :, None, None, :] - \
                           xxyy[None, None, ::stride, ::stride, :]
                    diff = (diff * diff).sum(4).sqrt()
                    diff = diff.pow(pow)
                # ----------------------------------------
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
                        num_diff = torch.sum((grad_smcorr - d_smcorr[0])) / d_smcorr[0].max()
                        print("local check: {:.4f}".format(num_diff))

                grad_smcorr = grad_smcorr.view(H, W, -1)
                f1_ = feats1_[b].view(C, H * W)
                f2_ = feats2_[b].view(C, H * W)
                timings["scale-feats"] += time.time() - tic

                tic = time.time()
                # BLock 2 ------------------------------------------
                # This is where the memory usage gets serious
                corr = torch.matmul(f1_.t(), f2_)
                timings["correlation"] += time.time() - tic

                # Direct backward pass for softmax
                tic = time.time()
                corr = corr.view(H, W, -1)
                smcorr = F.softmax(corr, dim=2)
                smcorr = smcorr.view(corr.shape)
                sum_term = torch.sum(grad_smcorr * smcorr, dim=2, keepdim=True)
                grad_corr = smcorr * (grad_smcorr - sum_term)
                timings["softmax"] += time.time() - tic

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

                tic = time.time()
                """The main work is done by some fairly beefy MM ops to compute
                pairwise correlations:"""
                grad_corr = grad_corr.view(H * W, H * W)
                grad_f1 = torch.matmul(grad_corr, f2_.t()).t()
                grad_f2 = torch.matmul(f1_, grad_corr)
                timings["corr-back"] += time.time() - tic
                tic = time.time()

                # grad_f1_inner = None
                # grad_f2_inner = None
                # grad_f1_outer = None
                # grad_f2_outer = None

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
            tic = time.time()

            """Must clear up all intermediate structures to avoid an autograd
            implosion."""
            # if True:
                # del grad_loss_b
                # del b
                # del grad_f1
                # del grad_f2
                # del smcorr
                # del corr
                # del diff
                # del params
            timings["cleanup"] += time.time() - tic

            # del f2
            # del f1
            # del grad_smcorr
            # del batch_grid_u
            # del xxyy
            # del feats1
            # del feats2

            # xx = copy.deepcopy(locals())
            # for key, val in locals().items():
                # if hasattr(val, "shape"):
                    # print("{:<15s}: {}".format(key, val.shape))
                # else:
                    # print("{}".format(key))

            timings["minibatch"] = time.time() - batch_tic
            print("==============")
            total_ratios = 0
            for key in timings:
                ratio = 100 * timings[key] / timings["minibatch"]
                print("{:.3f} ({:.2f}%) >>> {}".format(timings[key], ratio, key))
                total_ratios += ratio
            msg = "{:.3f}s >>> ratio total {}"
            print(msg.format(timings["minibatch"], total_ratios - 100))
            print("==============")


        return (grad_feats1, grad_feats2, grad_xxyy, grad_batch_u,
                grad_stride, grad_pow)


def dense_corr_check():
    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    dense_corr = DenseCorr.apply
    evc_dim = 4
    B, C, H, W = 4, evc_dim, 4, 4
    stride = 2
    feats1 = torch.randn(B, C, H, W, dtype=torch.double, requires_grad=True)
    feats2 = torch.randn(B, C, H, W, dtype=torch.double, requires_grad=True)
    batch_grid_u = torch.randn(B, H, W, 2, dtype=torch.double, requires_grad=False)

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
    print(test)


def main():
    parser = argparse.ArgumentParser(description="Gradient Check")
    parser.add_argument(
        "--func_name",
        default="dense_corr",
        choices=["activity-net", "LSMDC", "MSR-VTT"],
        type=str,
        help="The anme of the dataset to be processed",
    )
    args = parser.parse_args()

    if args.func_name == "dense_corr":
        dense_corr_check()
    elif args.func_name == "linear":
        linear_check()

if __name__ == "__main__":
    main()
