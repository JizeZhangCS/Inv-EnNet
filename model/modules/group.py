import torch
import torch.nn as nn


class GroupLayer(nn.Module):
    def __init__(self, factor=2, reverse=False, channel_num=3, split_point=1):
        super().__init__()
        self.factor = factor
        self.square_sf = factor*factor
        self.reverse = reverse
        self.split_point = split_point
        self.channel_num = channel_num

    def _group(self, x):
        assert x.shape[1] == self.square_sf * self.channel_num
        x_a_list = []
        x_b_list = []

        for i in range(self.channel_num):
            channel_a = x[:, i * self.square_sf:i * self.square_sf + self.split_point, :, :]
            channel_b = x[:, i * self.square_sf + self.split_point:(i + 1) * self.square_sf, :, :]
            if len(channel_a.shape) == 3:
                channel_a = torch.unsqueeze(channel_a, dim=1)
            if len(channel_b.shape) == 3:
                channel_b = torch.unsqueeze(channel_b, dim=1)
            x_a_list.append(channel_a)
            x_b_list.append(channel_b)

        return torch.cat(x_a_list, 1), torch.cat(x_b_list, 1)

    def _ungroup(self, tupl):
        x1, x2 = tupl
        assert x1.shape[1] == self.channel_num * self.split_point
        assert x2.shape[1] + x1.shape[1] == self.channel_num * self.square_sf
        channel_list = []
        b_channel_length = self.square_sf - self.split_point

        for i in range(self.channel_num):
            channel_a = x1[:, i * self.split_point:(i + 1) * self.split_point, :, :]
            channel_b = x2[:, i * b_channel_length:(i + 1) * b_channel_length, :, :]
            if len(channel_a.shape) == 3:
                channel_a = torch.unsqueeze(channel_a, dim=1)
            if len(channel_b.shape) == 3:
                channel_b = torch.unsqueeze(channel_b, dim=1)
            channel_list.append(channel_a)
            channel_list.append(channel_b)

        return torch.cat(channel_list, 1)

    def forward(self, input, rev=False, test_mode=None):
        if self.reverse:
            rev = not rev

        if not rev:
            output = self._group(input)  # Squeeze in forward
        else:
            output = self._ungroup(input)

        return output
