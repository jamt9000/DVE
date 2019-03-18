import torch
import torch.nn as nn
import torch.nn.functional as F


class IntermediateKeypointPredictor(nn.Module):

    def __init__(self, descriptor_dimension, num_annotated_points, num_intermediate_points, softargmax_mul=50.):
        super(IntermediateKeypointPredictor, self).__init__()
        self.nA = num_annotated_points
        self.nI = num_intermediate_points
        self.descriptor_dimension = descriptor_dimension
        self.softargmax_mul = softargmax_mul

        self.descriptors = nn.Parameter(torch.randn(descriptor_dimension,
                                                    num_annotated_points,
                                                    num_intermediate_points))

        self.linear = nn.Linear(num_intermediate_points*2, 2, bias=False)

    def forward(self, input):
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

            #f1 = F.normalize(f1, p=2, dim=0) * 20
            #f2 = F.normalize(f2, p=2, dim=0) * 20

            corr = torch.matmul(f1.t(), f2)

            smcorr = F.softmax(self.softargmax_mul * corr, dim=1)
            smcorr = smcorr.reshape(self.nA, self.nI, H, W)

            xpred = (smcorr * xx.view(1, 1, H, W)).sum(dim=(2, 3)) / smcorr.sum(dim=(2, 3))
            ypred = (smcorr * yy.view(1, 1, H, W)).sum(dim=(2, 3)) / smcorr.sum(dim=(2, 3))

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

        pred = self.linear(intermediate.reshape(B,self.nA,-1)).reshape(B,self.nA,2)

        return pred, intermediate


if __name__ == '__main__':
    m = IntermediateKeypointPredictor(4, 5, 10)
    o = m.forward(torch.randn(10, 4, 80, 75))
    print(o)
