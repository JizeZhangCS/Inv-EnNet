import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from base import BaseModel
from model.modules.spatial_flow_step_img_as_ft import SpatialFlowStepIAF
from model.modules.group import GroupLayer
from model.modules.subnet_constructor import subnet
from model.modules.squeeze import NoInvConvSqueezeLayer as SqueezeLayer


class InvEnNet(BaseModel):
    def __init__(self, channel_num=3, subnet_constructor=subnet('GammaConv'), block_size=8, split_point=2):
        super(InvEnNet, self).__init__()
        operations = []

        operations.append(SqueezeLayer(reverse=False))
        operations.append(GroupLayer(reverse=False, channel_num=channel_num, split_point=split_point))

        for i in range(block_size):
            operations.append(SpatialFlowStepIAF(subnet_constructor, channel_num=channel_num, split_point=split_point))

        operations.append(GroupLayer(reverse=True, channel_num=channel_num, split_point=split_point))
        operations.append(SqueezeLayer(reverse=True))

        self.operations = nn.ModuleList(operations)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)



    def test_forward(self, x, rev=False, output_step=False):
        shift_list = []
        scale_list = []

        ft = x

        if not rev:
            for op in self.operations:
                ret_value = op.forward(ft, rev, test_mode=True)
                if isinstance(ret_value[0], tuple):
                    ft, shift, scale = ret_value
                    shift_list.append(F.interpolate(shift, scale_factor=2, mode='nearest'))
                    scale_list.append(F.interpolate(scale, scale_factor=2, mode='nearest'))
                else:
                    ft = ret_value
        else:
            for op in reversed(self.operations):
                ret_value = op.forward(ft, rev, test_mode=True)
                if isinstance(ret_value[0], tuple):
                    ft, shift, scale = ret_value
                    shift_list.append(F.interpolate(shift, scale_factor=2, mode='nearest'))
                    scale_list.append(F.interpolate(scale, scale_factor=2, mode='nearest'))
                else:
                    ft = ret_value

        output_list = [x]
        for shift, scale in zip(shift_list, scale_list):
            output_list.append(output_list[-1]*scale+shift)

        return output_list if output_step else output_list[-1]

    def forward(self, x, rev=False, output_step=False, test_scale_shift=False):
        if test_scale_shift:
            return self.test_forward(x, rev, output_step)

        loss = 0
        if not output_step:
            out = x  # x: [N,3,H,W]
            if not rev:
                for op in self.operations:
                    ret_value = op.forward(out, rev)
                    if isinstance(ret_value[0], tuple):
                        out, loss_curr = ret_value
                        loss += loss_curr
                    else:
                        out = ret_value

            else:
                for op in reversed(self.operations):
                    ret_value = op.forward(out, rev)
                    if isinstance(ret_value[0], tuple):
                        out, loss_curr = ret_value
                        loss += loss_curr
                    else:
                        out = ret_value
        else:
            out = [x]
            if not rev:
                for op in self.operations:
                    ret_value = op.forward(out[-1], rev)
                    if isinstance(ret_value[0], tuple):
                        out.append(ret_value[0])
                        loss += ret_value[1]
                    else:
                        out.append(ret_value)
            else:
                for op in reversed(self.operations):
                    ret_value = op.forward(out[-1], rev)
                    if isinstance(ret_value[0], tuple):
                        out.append(ret_value[0])
                        loss += ret_value[1]
                    else:
                        out.append(ret_value)

        return out, loss
