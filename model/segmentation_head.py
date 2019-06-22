import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    def __init__(self, descriptor_dimension, num_classes):
        super().__init__()
        self.descriptor_dimension = descriptor_dimension
        self.classifier = nn.Conv2d(
            in_channels=descriptor_dimension,
            out_channels=num_classes,
            kernel_size=1,
            bias=True,
        )

    def forward(self, input):
        return self.classifier(input[0].detach())


if __name__ == '__main__':
    desc_dim = 16
    mod = SegmentationHead(desc_dim, num_classes=10)
    x = [torch.randn(10, desc_dim, 80, 75)]
    with torch.no_grad():
        segs = mod.forward(x)
