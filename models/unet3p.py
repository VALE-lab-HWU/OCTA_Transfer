import torch
from torch import nn
from .unet import Unet
from .layers.convBlock import ConvBlock, ConvBlockOne,  DownConvBlockMax, \
    ConConvBlockUp, ConConvBlockDown, ConConvBlockOneUp, ConConvBlockOneDown
from functools import partial
# import numpy as np


class Unet3Plus(Unet):
    def __init__(
            self, connection_block_fn_up, connection_block_fn_down,
            connection_block_fn, *args, n_block=4, init_features=64, **kwargs):
        self.scales = (n_block + 1) * init_features
        self.connection_block_fn = connection_block_fn
        self.connection_block_fn_up = connection_block_fn_up
        self.connection_block_fn_down = connection_block_fn_down
        super(Unet3Plus, self).__init__(
            *args, n_block=n_block, init_features=init_features, **kwargs)
        self._make_connections()
        self.initialize_weights()

    def _make_connections(self):
        feature = self.init_features
        res_all = nn.ModuleDict()
        for i in range(0, self.n_block+1):
            res = nn.ModuleDict()
            f = feature * 2 ** i
            for j in range(0, self.n_block):
                sc = 2 ** abs(j-i)
                if i == j:
                    block = self.connection_block_fn(f, feature)
                elif i > j:
                    if i == self.n_block:
                        block = self.connection_block_fn_up(f, feature, sc=sc)
                    else:
                        block = self.connection_block_fn_up(
                            self.scales, feature, sc=sc)
                else:
                    block = self.connection_block_fn_down(f, feature, sc=sc)
                res[f'con_{i}_{j}'] = block
            res_all[f'con_{i}'] = res
        self.connections = res_all

    def _make_expanding_path(self):
        res = []
        for i in range(self.n_block):
            block = self.expand_block_fn(self.scales, self.scales)
            res.append(block)
        self.expanding_path = nn.ModuleList(res)

    def _make_last(self):
        self.last = nn.Conv2d(self.scales,
                              self.out_channels,
                              kernel_size=1)
        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        xt = []
        for block in self.contracting_path:
            x, tmp = block(x)
            xt.append(tmp)

        y = self.bottleneck(x)

        xp = []
        for i, name in enumerate(self.connections):
            for j, name_2 in enumerate(self.connections[name]):
                if j >= i:
                    layer = self.connections[name][name_2]
                    tmp = layer(xt[i])
                    if len(xp) == j:
                        xp.append([])
                    xp[j].append(tmp)

        yt = [y]
        j = self.n_block - 1
        for i, block in enumerate(self.expanding_path):
            tmp_y = []
            for k, cons in enumerate(self.connections.values()):
                if k > j:
                    val = yt[self.n_block-k]
                    layer = cons[f'con_{k}_{j}']
                    tmp_y.append(layer(val))
            y = block(torch.cat([*xp[j], *tmp_y], dim=1))
            j -= 1
            yt.append(y)

        return self.out(self.last(y))


class ResidualUnet3Plus(Unet3Plus):
    def __init__(self, bn=True, **kwargs):
        block_fn = partial(ConvBlock, residual=True, bn=True)
        downblock_fn = partial(DownConvBlockMax, residual=True, bn=True)
        upconblock_fn = partial(ConConvBlockUp, residual=True, bn=True)
        downconblock_fn = partial(ConConvBlockDown, residual=True, bn=True)
        super(ResidualUnet3Plus, self).__init__(
            upconblock_fn, downconblock_fn, block_fn,
            downblock_fn, block_fn, block_fn,
            **kwargs)


class ClassicalUnet3Plus(Unet3Plus):
    def __init__(self, bn=True, **kwargs):
        block_fn = partial(ConvBlock, residual=False, bn=False)
        botblock_fn = partial(ConvBlock, residual=False, bn=True)
        downblock_fn = partial(DownConvBlockMax, residual=False, bn=True)
        upconblock_fn = partial(ConConvBlockUp, residual=False, bn=False)
        downconblock_fn = partial(ConConvBlockDown, residual=False, bn=False)
        super(ClassicalUnet3Plus, self).__init__(
            upconblock_fn, downconblock_fn, block_fn,
            downblock_fn, block_fn, botblock_fn,
            **kwargs)


class PaperUnet3Plus(Unet3Plus):
    def __init__(self, bn=True, **kwargs):
        block_fn = partial(ConvBlock, residual=False, bn=True)
        block_fn_3p = partial(ConvBlockOne, residual=False, bn=True)
        downblock_fn = partial(DownConvBlockMax, residual=False, bn=True)
        upconblock_fn = partial(ConConvBlockOneUp, residual=False, bn=True)
        downconblock_fn = partial(ConConvBlockOneDown, residual=False, bn=True)
        super(PaperUnet3Plus, self).__init__(
            upconblock_fn, downconblock_fn, block_fn_3p,
            downblock_fn, block_fn_3p, block_fn,
            **kwargs)
