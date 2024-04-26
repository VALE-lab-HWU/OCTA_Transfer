import torch
from torch import nn
from .unet import Unet
from .layers.convBlock import ConvBlock, UpConvBlockConv, DownConvBlockMax
from functools import partial


class UnetPlusPlus(Unet):
    def __init__(self, connection_block_fn, *args, **kwargs):
        super(UnetPlusPlus, self).__init__(*args, **kwargs)
        self.connection_block_fn = connection_block_fn
        self._make_connections()
        self.initialize_weights()

    def _make_connections(self):
        feature = self.init_features
        res_all = nn.ModuleDict()
        for i in range(0, self.n_block-1):
            res = nn.ModuleDict()
            f = feature * 2 ** i
            for j in range(1, self.n_block-i):
                # in, out, mid
                block = self.connection_block_fn(f * (j+1), f,
                                                 mid_channels=f * 2, sc=2)
                # up = nn.Upsample(scale_factor=2, mode='bilinear',
                #                  align_corners=True)
                res[f'con_{i}_{j}'] = block
            res_all[f'con_{i}'] = res
        self.connections = res_all

    def _make_expanding_path(self):
        feature = self.init_features  # it's shorter
        res = []
        for i in range(self.n_block):
            ri = self.n_block - i - 1
            nf = 2 + i
            # in, out, mid
            block = self.expand_block_fn(
                nf * (feature * 2 ** (ri)),
                feature * 2 ** (ri),
                mid_channels=feature * 2 ** (ri+1),
            )
            res.append(block)
        self.expanding_path = nn.ModuleList(res)

    def forward(self, x):
        xt = []
        for block in self.contracting_path:
            x, tmp = block(x)
            xt.append(tmp)

        y = self.bottleneck(x)

        xp = []
        for i, name in enumerate(self.connections):
            tmp = []
            for j, layer in enumerate(self.connections[name].values()):
                tmp_cat = torch.cat([xt[i], *tmp], dim=1)
                tmp.append(layer(xt[i+1], tmp_cat))
            xp.append(tmp)

        yt = []
        j = self.n_block - 1
        for i, block in enumerate(self.expanding_path):
            tmpy = torch.cat([
                xt[-(i+1)],
                *(xp[j] if len(xp) > j else torch.tensor([]))], dim=1)
            y = block(y, tmpy)
            j -= 1
            yt.append(y)  # not needed yet

        return self.out(self.last(y))


class ClassicalUnetPlusPlus(UnetPlusPlus):
    def __init__(self, bn=True, **kwargs):
        upblock_fn = partial(UpConvBlockConv, residual=False, bn=False)
        downblock_fn = partial(DownConvBlockMax, residual=False, bn=True)
        block_fn = partial(ConvBlock, residual=False, bn=True)
        super(ClassicalUnetPlusPlus, self).__init__(
            upblock_fn, downblock_fn, upblock_fn, block_fn,
            **kwargs)


class ResidualUnetPlusPlus(UnetPlusPlus):
    def __init__(self, **kwargs):
        upblock_fn = partial(UpConvBlockConv, residual=True, bn=True)
        downblock_fn = partial(DownConvBlockMax, residual=True, bn=True)
        block_fn = partial(ConvBlock, residual=True, bn=True)
        super(ResidualUnetPlusPlus, self).__init__(
            upblock_fn, downblock_fn, upblock_fn, block_fn
        )
