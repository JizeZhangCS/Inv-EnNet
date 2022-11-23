import torch.nn as nn


def squeeze2d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    assert H % factor == 0 and W % factor == 0, "{}".format((H, W, factor))
    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)
    return x


def unsqueeze2d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    factor2 = factor ** 2
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    assert C % (factor2) == 0, "{}".format(C)
    x = input.view(B, C // factor2, factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor2), H * factor, W * factor)
    return x


class NoInvConvSqueezeLayer(nn.Module):
    def __init__(self, factor=2, reverse=False):
        super().__init__()
        self.factor = factor
        self.reverse = reverse

    def forward(self, input, rev=False, test_mode=None):
        if self.reverse:
            rev = not rev

        if not rev:
            output = squeeze2d(input, self.factor)  # Squeeze in forward
        else:
            output = unsqueeze2d(input, self.factor)

        return output