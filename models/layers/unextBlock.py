# own implementation of the convnext block from the convnext paper
# doi.org/10.48550/arXiv.2201.03545
# original code:
# https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
#
# layer scale:
# doi.org/10.48550/arXiv.2103.17239
# stochastic depth:
# doi.org/10.48550/arXiv.1603.09382

import torch
import torch.nn as nn
from .layerNorm import LayerNorm
from .downBlock import DownConvGeneric
from timm.models.layers import DropPath


class NextBlock(nn.Module):
    def __init__(self, in_channels, out_channels,  # out_c to stay consistent
                 drop_path=0., layer_scale=1e6):
        super(NextBlock, self).__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 7),
                                 padding=(3, 3), groups=in_channels)
        self.norm = LayerNorm(in_channels)
        # could be 1x1 conv
        self.pw_conv1 = nn.Linear(in_channels, 4 * in_channels)
        self.activation = nn.GELU()
        # could be 1x1 conv
        self.pw_conv2 = nn.Linear(4 * in_channels, in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()  # Stochastic Depth
        self.gamma = nn.Parameter(
            layer_scale * torch.ones((in_channels)), requires_grad=True) \
            if layer_scale > 0 else None  # Layer Scale

    def forward(self, x):  # b, c, h, w
        tmp = x
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1)  # b, h, w, c
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.activation(x)
        x = self.pw_conv2(x)
        x = (x * self.gamma) if self.gamma is not None else x  # Layer Scale
        x = x.permute(0, 3, 1, 2)  # b, c, h, w
        x = tmp + self.drop_path(x)  # stochastic depth
        return x


class DownsampleNextBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(DownsampleNextBlock, self).__init__()
        self.add_module('norm', LayerNorm(in_channels, eps=1e-6,
                                          data_format="channels_first"))
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=2, stride=2)


class UpsampleNextBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(UpsampleNextBlock, self).__init__()
        self.add_module('norm', LayerNorm(in_channels, eps=1e-6,
                                          data_format="channels_first"))
        self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                       kernel_size=2, stride=2)


class StemBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(StemBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=5, stride=1, padding=2)
        self.add_module('norm', LayerNorm(in_channels, eps=1e-6,
                                          data_format="channels_first"))


class UpNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_block=1):
        super().__init__()
        seq = []
        for i in range(n_block):
            seq.append(NextBlock(in_channels, out_channels))
        self.conv = nn.Sequential(*seq)
        self.up = UpsampleNextBlock(in_channels, out_channels)
        self.project = nn.Conv2d(in_channels, out_channels,
                                 kernel_size=1, stride=1)

    def forward(self, x, skip_x):
        up = self.up(x)
        return self.project(self.conv(torch.cat([up, skip_x], dim=1)))


class DownNextBlock(DownConvGeneric):
    def __init__(self, in_channels, out_channels,
                 drop_path=0., layer_scale=1e6, n_block=1):
        seq = []
        for i in range(n_block):
            seq.append(NextBlock(in_channels, out_channels,
                                 drop_path=drop_path, layer_scale=layer_scale))
        conv = nn.Sequential(*seq)
        down = DownsampleNextBlock(in_channels, out_channels)
        super().__init__(down, conv)
