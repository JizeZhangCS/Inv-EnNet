import torch
import torch.nn as nn
from .invconv import InvertibleConv1x1


class FlowStep(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=0.8):
        super(FlowStep, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        # self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

        in_channels = 3
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)

    def forward(self, x, rev=False):
        if not rev:
            # invert1x1conv
            x, logdet = self.invconv(x, logdet=0, reverse=False)

            # split to 1 channel and 2 channel.
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
            y1 = x1
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
            out = torch.cat((y1, y2), 1)
        else:
            # split.
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1

            x = torch.cat((y1, y2), 1)

            # inv permutation
            out, logdet = self.invconv(x, logdet=0, reverse=True)

        return out
