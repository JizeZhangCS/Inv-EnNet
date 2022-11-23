import torch
import torch.nn as nn


class SpatialFlowStep(nn.Module):
    def __init__(self, subnet_constructor, channel_num=3, squeeze_factor=2, clamp=0.8, split_point=1):
        super(SpatialFlowStep, self).__init__()

        self.clamp = clamp
        self.square_sf = squeeze_factor*squeeze_factor
        self.channel_num = channel_num
        self.split_point = split_point

        self.G = subnet_constructor(split_point*channel_num, channel_num)
        self.H = subnet_constructor(split_point*channel_num, channel_num)
        self.G_prime = subnet_constructor((self.square_sf-split_point)*channel_num, channel_num)
        self.H_prime = subnet_constructor((self.square_sf-split_point)*channel_num, channel_num)

    def _expand(self, x, repeats):
        if repeats != 1:
            x = x.repeat_interleave(repeats, dim=1)
        return x

    def forward(self, input_tupl, rev=False, test_mode=False):
        if not rev:
            x1, x2 = input_tupl
            scale = torch.exp(self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1))
            shift = self.G(x1)
            y2 = x2.mul(self._expand(scale, repeats=self.square_sf-self.split_point)) + \
                 self._expand(shift, repeats=self.square_sf-self.split_point)  # 2 channel
            # now enhance x1 to y1
            scale_prime = torch.exp(self.clamp * (torch.sigmoid(self.H_prime(y2)) * 2 - 1))
            shift_prime = self.G_prime(y2)
            y1 = x1.mul(self._expand(scale_prime, repeats=self.split_point)) + \
                 self._expand(shift_prime, repeats=self.split_point)
            out = (y1, y2)
        else:
            y1, y2 = input_tupl
            # degrade y1 to x1
            scale_prime = torch.exp(self.clamp * (torch.sigmoid(self.H_prime(y2)) * 2 - 1))
            shift_prime = self.G_prime(y2)
            x1 = (y1 - self._expand(shift_prime, repeats=self.split_point)).div(
                self._expand(scale_prime, repeats=self.split_point))
            # now degrade y2 to x2
            scale = torch.exp(self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1))
            shift = self.G(x1)
            x2 = (y2 - self._expand(shift, repeats=self.square_sf-self.split_point)).div(
                self._expand(scale, repeats=self.square_sf-self.split_point))
            out = (x1, x2)

        # self.utest(shift)
        # self.utest(scale)
        if not test_mode:
            return out
        return out, shift, scale
