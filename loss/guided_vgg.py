import torch
from .modules.vgg_feature import vgg_ft_calc


class GuidedPerceptualLoss(torch.nn.Module):
    def __init__(self, last_only=False):
        super(GuidedPerceptualLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.last_only = last_only

    def forward(self, input, target):
        input_ft_list, target_ft_list = vgg_ft_calc(input, target, guided=True)
        if self.last_only:
            return self.mse_loss(input_ft_list[-1], target_ft_list[-1])

        loss = 0
        for idx, (input_ft, target_ft) in enumerate(zip(input_ft_list, target_ft_list)):
            loss += self.mse_loss(input_ft, target_ft) / 4

        return loss
