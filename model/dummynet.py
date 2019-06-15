import torch
from base import BaseModel


class DummyNet(BaseModel):
    def __init__(self, num_output_channels):
        super(DummyNet, self).__init__()
        self.num_output_channels = num_output_channels

    def forward(self, x):
        msg = "expected {} output channels, found {}"
        msg = msg.format(x.shape[1], self.num_output_channels)
        assert x.shape[1] == self.num_output_channels, msg
        return [x]


if __name__ == '__main__':
    x = torch.randn(2, 3, 70, 70)
    net = DummyNet()
    y = net.forward(x)
    print(y[0].shape)
