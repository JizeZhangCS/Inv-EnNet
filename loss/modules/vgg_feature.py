import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .guided_filter import GuidedFilter


class VGGFeature(torch.nn.Module):
    def __init__(self, r=28, download=True):
        super(VGGFeature, self).__init__()
        vgg_model = models.vgg16(pretrained=download)
        if not download:
            pre = torch.load("./vgg16-397923af.pth")
            vgg_model.load_state_dict(pre)

        blocks = []
        blocks.append(vgg_model.features[:4].eval())
        blocks.append(vgg_model.features[4:9].eval())
        blocks.append(vgg_model.features[9:16].eval())
        blocks.append(vgg_model.features[16:23].eval())
        for block in blocks:
            for layer in block:
                layer.requires_grad = False
        self.vgg = nn.ModuleList(blocks)
        self.guided_filter = GuidedFilter(r)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # self.mse_loss = torch.nn.MSELoss()

    def forward(self, input, output, each_layer=True, guided=True):
        mean = self.mean.to(input.device)
        std = self.std.to(input.device)
        self.vgg = self.vgg.to(input.device)

        # x = F.interpolate(input, size=(224, 224), mode='bilinear', align_corners=False)
        # y = F.interpolate(output, size=(224, 224), mode='bilinear', align_corners=False)

        if guided:
            self.guided_filter = self.guided_filter.to(input.device)
            input = self.guided_filter(input, output)
            input = torch.clamp(input, 0, 1)

        input = (input-mean) / std
        output = (output-mean) / std

        input_list = [input]
        output_list = [output]
        for i, block in enumerate(self.vgg):
            input_list.append(block(input_list[-1]))
            output_list.append(block(output_list[-1]))

        return (input_list[1:], output_list[1:]) if each_layer else (input_list[-1], output_list[-1])


vgg_ft_calc = VGGFeature()
