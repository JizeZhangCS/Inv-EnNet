import torch
import torch.nn as nn
import torch.nn.functional as F
import model.modules.module_util as mutil
# from .act_norm import Conv2d, Conv2dZeros


# class GammaConvWithActnorm(nn.Module):
#     def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
#         super(GammaConvWithActnorm, self).__init__()
#         self.conv1 = Conv2d(channel_in, gc, 3)
#         self.conv2 = Conv2d(gc, gc, 1)
#         self.conv3 = Conv2d(gc, channel_out, 3)
#         self.relu = nn.ReLU(inplace=True)
#         self.gamma = nn.Parameter(torch.Tensor([1]))
#
#         if init == 'xavier':
#             mutil.initialize_weights_xavier([self.conv1, self.conv2], 0.1)
#         else:
#             mutil.initialize_weights([self.conv1, self.conv2], 0.1)
#         mutil.initialize_weights(self.conv3, 0)
#
#     def forward(self, x):   # x: [B, channel_num/2, H, W]
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.conv3(x) * self.gamma
#         return x


class GammaConv(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(GammaConv, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(gc, gc, 1, bias=bias)
        self.conv3 = nn.Conv2d(gc, channel_out, 3, padding=1, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.gamma = nn.Parameter(torch.Tensor([1]))

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2], 0.1)
        mutil.initialize_weights(self.conv3, 0)

    def forward(self, x):   # x: [B, channel_num/2, H, W]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x) * self.gamma
        return x


# class GlobalConv(nn.Module):
#     def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
#         super(GlobalConv, self).__init__()
#         self.conv1 = nn.Conv2d(channel_in, gc, kernel_size=7, stride=2, bias=bias)
#         self.conv2 = nn.Conv2d(gc, gc, kernel_size=5, stride=2, bias=bias)
#         self.conv3 = nn.Conv2d(gc, channel_out, kernel_size=3, stride=2, bias=bias)
#         self.relu = nn.ReLU(inplace=True)
#         self.glob_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
#
#         if init == 'xavier':
#             mutil.initialize_weights_xavier([self.conv1, self.conv2], 0.1)
#         else:
#             mutil.initialize_weights([self.conv1, self.conv2], 0.1)
#         mutil.initialize_weights(self.conv3, 0)
#
#     def forward(self, x):   # x: [B, 1, H, W]
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#         x = self.glob_avg_pooling(x)
#         return x            # x: [B, 2, H, W]


class SubnetCore(nn.Module):
    def __init__(self, net_structure, channel_in, channel_out, init='xavier'):
        super(SubnetCore, self).__init__()
        self.net_structure = net_structure

        if net_structure == 'DBNet':
            if init == 'xavier':
                self.subnet_core = DenseBlock(channel_in, channel_out, init)
            else:
                self.subnet_core = DenseBlock(channel_in, channel_out)
        elif net_structure == 'GammaConv':
            self.subnet_core = GammaConv(channel_in, channel_out, init)
        elif net_structure == 'GammaConvWithActnorm':
            self.subnet_core = GammaConvWithActnorm(channel_in, channel_out, init)
        elif net_structure == 'GlobalConv':
            self.subnet_core = GlobalConv(channel_in, channel_out, init)
        else:
            raise NotImplementedError()

    # def _init_local_consistancy(self, wt_num, curr_level):
    #     # shrink_c_time = curr_level
    #     # shrink_wh_time = wt_num-curr_level
    #     shrinker = []
    #     for i in range(wt_num-curr_level):
    #         shrinker.append(nn.MaxPool2d(kernel_size=2))
    #     for i in range(curr_level):
    #         in_channels = 4**(curr_level-i)
    #         out_channels = in_channels//4
    #         shrinker.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))
    #     self.shrinker = nn.ModuleList(shrinker)

        # def _expander(wt_num, curr_level):
        #     def expand_operation(x):
        #         for i in range(curr_level):
        #             x = x.repeat_interleave(4, dim=1)
        #         for i in range(wt_num - curr_level):
        #             x = F.interpolate(x, scale_factor=2, mode='nearest')
        #         return x
        #     return expand_operation
        # self.expander = _expander(wt_num, curr_level)

    def forward(self, x):
        # x: [B, 1, H, W]
        return self.subnet_core(x)

        # if self.net_structure != 'GammaConv':
        #     raise NotImplementedError("only GammaConv can be used as local consistancy")
        #
        # self._init_local_consistancy(wt_num, curr_level)
        # self.shrinker = self.shrinker.to(x.device)
        # for block in self.shrinker:
        #     x = block(x)
        # x = self.subnet_core(x)
        # x = self.expander(x)
        # return x


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        return SubnetCore(net_structure, channel_in, channel_out, init)

    return constructor
