from torch import nn
from .unet import Unet
from .layers.csnetBlock import AffinityAttention
from .layers.convBlock import ConvBlock, DownConvBlockMax, UpConvBlockConv
from functools import partial
# import numpy as np


class ResidualCSnet(Unet):
    def __init__(self, *args, **kwargs):
        upblock_fn = partial(UpConvBlockConv, residual=True, bn=True)
        downblock_fn = partial(DownConvBlockMax, residual=True, bn=True)
        super().__init__(
            contract_block_fn=downblock_fn,
            expand_block_fn=upblock_fn,
            bottleneck_block_fn=ResidualCSnet.get_bottleneck,
            *args, **kwargs)

    def get_bottleneck(in_channels, out_channels):
        return nn.Sequential(
            ConvBlock(in_channels, out_channels, residual=True, bn=True),
            AffinityAttention(out_channels)
        )


class ClassicalCSnet(Unet):
    def __init__(self, bn=True, *args, **kwargs):
        upblock_fn = partial(UpConvBlockConv, residual=False, bn=False)
        downblock_fn = partial(DownConvBlockMax, residual=False, bn=True)
        super().__init__(
            contract_block_fn=downblock_fn,
            expand_block_fn=upblock_fn,
            bottleneck_block_fn=ClassicalCSnet.get_bottleneck,
            *args, **kwargs)

    def get_bottleneck(in_channels, out_channels):
        return nn.Sequential(
            ConvBlock(in_channels, out_channels, residual=False, bn=True),
            AffinityAttention(out_channels)
        )
