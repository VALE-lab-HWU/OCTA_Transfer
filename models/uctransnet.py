from torch import nn
from .unet import Unet
from .layers.uctransnetMTCBlock import DefaultMTC
from .layers.uctransnetUpBlock import UpConvBlockCCAConv, UpConvBlockCCAUp
from .layers.convBlock import ConvBlock, DownConvBlockMax, DownConvBlockOneMax
from functools import partial


class UCTransNet(Unet):
    def __init__(self, *args, img_size=400, **kwargs):
        super(UCTransNet, self).__init__(*args, **kwargs)
        self.mtc = DefaultMTC(img_size=img_size)

    def forward(self, x):
        xt = []
        for block in self.contracting_path:
            x, tmp = block(x)
            xt.append(tmp)

        *xt, _ = self.mtc(*xt)

        y = self.bottleneck(x)

        for i, block in enumerate(self.expanding_path):
            y = block(y, xt[-(i+1)])

        return self.out(self.last(y))


class ClassicalUCTransNet(UCTransNet):
    def __init__(self, img_size=400, **kwargs):
        block_fn = partial(ConvBlock, residual=False, bn=True)
        downblock_fn = partial(DownConvBlockMax, residual=False, bn=True)
        upblock_fn = partial(UpConvBlockCCAConv, residual=False, bn=False)
        super(ClassicalUCTransNet, self).__init__(
            downblock_fn, upblock_fn, block_fn, img_size=img_size, **kwargs)


class ResidualUCTransNet(UCTransNet):
    def __init__(self, img_size=400, **kwargs):
        block_fn = partial(ConvBlock, residual=True, bn=True)
        downblock_fn = partial(DownConvBlockMax, residual=True, bn=True)
        upblock_fn = partial(UpConvBlockCCAConv, residual=True, bn=True)
        super(ResidualUCTransNet, self).__init__(
            downblock_fn, upblock_fn, block_fn, img_size=img_size, **kwargs)


class PaperUCTransNet(UCTransNet):
    def __init__(self, img_size=400, **kwargs):
        block_fn = partial(ConvBlock, residual=False, bn=True)
        downblock_fn = partial(DownConvBlockMax, residual=False, bn=True)
        downblock_fn_2 = partial(DownConvBlockOneMax, residual=False, bn=True)
        upblock_fn = partial(UpConvBlockCCAUp, residual=False, bn=True)
        self.contract_block_fn_2 = downblock_fn_2
        super(PaperUCTransNet, self).__init__(
            downblock_fn, upblock_fn, block_fn, img_size=img_size, **kwargs)

    def _make_contracting_path(self):
        feature = self.init_features
        res = []
        for i in range(-1, self.n_block - 1):
            if i == -1:
                block = self.contract_block_fn_2(self.in_channels,
                                                 feature)
            else:
                block = self.contract_block_fn(feature * 2 ** i,
                                               feature * 2 ** (i+1))
            res.append(block)
        self.contracting_path = nn.ModuleList(res)

    def _make_expanding_path(self):
        feature = self.init_features  # it's shorter
        res = []
        for i in range(self.n_block):
            ri = self.n_block - i - 2
            if i == self.n_block-1:
                block = self.expand_block_fn(feature * 2 ** (ri+2),
                                             feature * 2 ** (ri+1))
            else:
                block = self.expand_block_fn(feature * 2 ** (ri+2),
                                             feature * 2 ** (ri))
            res.append(block)
        self.expanding_path = nn.ModuleList(res)

    def _make_bottleneck(self):
        b = self.bottleneck_block_fn(
            self.init_features * 2 ** (self.n_block-1),
            self.init_features * 2 ** (self.n_block-1))
        self.bottleneck = b
