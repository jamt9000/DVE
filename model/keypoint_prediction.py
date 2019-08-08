import torch
import torch.nn as nn
import torch.nn.functional as F

# need to disable for evaluation to ensure compatibility with prior work
DENSE_CONV = False


class IntermediateKeypointPredictor(nn.Module):
    def __init__(self, descriptor_dimension, num_annotated_points,
                 num_intermediate_points, softargmax_mul=50., numerical_check=False):
        super().__init__()
        self.nA = num_annotated_points
        self.nI = num_intermediate_points
        self.descriptor_dimension = descriptor_dimension
        self.softargmax_mul = softargmax_mul

        weights = torch.randn(descriptor_dimension, self.nA, self.nI)
        linearlist = nn.ModuleList([
            nn.Linear(num_intermediate_points * 2, 2, bias=False)
            for l in range(self.nA)
        ])

        # use the same weights to maintain numerical equiv checks
        latent_dim = self.nA * self.nI
        self.inner_conv = nn.Conv2d(
            in_channels=descriptor_dimension,
            out_channels=latent_dim,
            kernel_size=1,
            bias=False,
        )
        weights = weights.view(descriptor_dimension, -1).t()
        weights = weights.view(latent_dim, descriptor_dimension, 1, 1)
        self.inner_conv.weight.data = weights
        self.latent_dim = latent_dim

        if DENSE_CONV:
            self.reg_conv = nn.Conv2d(
                in_channels=latent_dim * 2,
                out_channels=2 * self.nA,
                kernel_size=1,
                bias=False,
            )
        else:  # use groups to reproduce the independent virtual keypoint usage
            self.reg_conv = nn.Conv2d(
                in_channels=latent_dim * 2,
                out_channels=2 * self.nA,
                kernel_size=1,
                groups=self.nA,
                bias=False,
            )
            reg_weights = torch.zeros(2 * self.nA, self.nI * 2)
            for ii in range(0, self.nA):
                reg_weights[2 * ii:2 * (ii + 1)] = linearlist[ii].weight.data
            self.reg_conv.weight.data = reg_weights.view(2 * self.nA, -1, 1, 1)

        # remove unused elements
        self.descriptors = nn.Parameter(weights)
        self.linear = linearlist

    def forward(self, input):
        input = input[0].detach()
        B, C, H, W = input.shape

        assert self.descriptor_dimension == C
        xi = torch.linspace(-1, 1, W, device=input.device)
        yi = torch.linspace(-1, 1, H, device=input.device)
        yy, xx = torch.meshgrid(yi, xi)

        corr = self.inner_conv(input)
        corr = corr.view(B, self.latent_dim, H * W)
        smcorr = F.softmax(self.softargmax_mul * corr, dim=2)
        smcorr = smcorr.reshape(B, self.nA, self.nI, H, W)

        mass = smcorr.sum(dim=(3, 4))
        xpred = (smcorr * xx.view(1, 1, 1, H, W)).sum(dim=(3, 4)) / mass
        ypred = (smcorr * yy.view(1, 1, 1, H, W)).sum(dim=(3, 4)) / mass
        intermediate = torch.stack((xpred, ypred), dim=3)
        # pred = [
        #     self.linear[i](intermediate[:, i, :, :].reshape(B, -1)).reshape(B, 1, 2)
        #     for i in range(self.nA)
        # ]
        # pred = torch.cat(pred, 1)
        pred = self.reg_conv(intermediate.view(B, -1, 1, 1)).view(B, self.nA, 2)
        return pred, intermediate

    def forward_orig(self, input):
        input = input[0].detach()
        B, C, H, W = input.shape

        assert self.descriptor_dimension == C

        xi = torch.linspace(-1, 1, W, device=input.device)
        yi = torch.linspace(-1, 1, H, device=input.device)
        yy, xx = torch.meshgrid(yi, xi)

        intermediate = torch.zeros(B, self.nA, self.nI, 2, device=input.device)

        for b in range(B):
            f1 = self.descriptors.reshape(C, -1)  # source
            f2 = input[b].reshape(C, H * W)  # target

            # f1 = F.normalize(f1, p=2, dim=0) * 20
            # f2 = F.normalize(f2, p=2, dim=0) * 20

            corr = torch.matmul(f1.t(), f2)

            smcorr = F.softmax(self.softargmax_mul * corr, dim=1)
            smcorr = smcorr.reshape(self.nA, self.nI, H, W)

            xpred = (smcorr * xx.view(1, 1, H, W)).sum(dim=(2, 3)) / smcorr.sum(dim=(2,
                                                                                     3))
            ypred = (smcorr * yy.view(1, 1, H, W)).sum(dim=(2, 3)) / smcorr.sum(dim=(2,
                                                                                     3))

            intermediate[b, :, :, 0] = xpred
            intermediate[b, :, :, 1] = ypred

            # for a in range(self.nA):
            #     for i in range(self.nI):
            #         real_argmax = torch.argmax(smcorr[a,i])
            #         rx = xx.reshape(-1)[real_argmax]
            #         ry = yy.reshape(-1)[real_argmax]
            #         sx = xpred[a,i]
            #         sy = ypred[a,i]
            #         print("[%d,%d] soft (%f,%f) real (%f,%f)" % (a,i,sx,sy,rx,ry))
        self.intermediate = intermediate

        pred = [
            self.linear[i](intermediate[:, i, :, :].reshape(B, -1)).reshape(B, 1, 2)
            for i in range(self.nA)
        ]
        pred = torch.cat(pred, 1)

        return pred, intermediate


if __name__ == '__main__':
    desc_dim = 16
    num_annotated_points = 5
    num_intermediate_points = 9
    m = IntermediateKeypointPredictor(
        desc_dim,
        num_annotated_points=num_annotated_points,
        num_intermediate_points=num_intermediate_points,
        numerical_check=True,
    )
    x = [torch.randn(10, desc_dim, 80, 75)]
    with torch.no_grad():
        o1, int1 = m.forward_orig(x)
        o2, int2 = m.forward(x)
    out_diff = o1 - o2
    int_diff = int1 - int2
    print("output diffs {}".format(out_diff))
    print("intermediate diffs {}".format(int_diff))
