from torch import nn
from .layers.convBlock import ConvBlock, UpConvBlockConv, DownConvBlockMax
from functools import partial


class Unet(nn.Module):
    def __init__(self, contract_block_fn, expand_block_fn, bottleneck_block_fn,
                 in_channels=1, out_channels=2,
                 init_features=64, n_block=4, **kwargs):
        super(Unet, self).__init__()
        self.init_features = init_features
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_block = n_block
        self.expand_block_fn = expand_block_fn
        self.contract_block_fn = contract_block_fn
        self.bottleneck_block_fn = bottleneck_block_fn
        self._make_layer()
        self.initialize_weights()

    def freeze(self, names, nc=True, grad=False):
        if nc:
            nc = dict(self.named_children())
        else:
            nc = dict(self.named_modules())
        for name in names:
            nc[name].requires_grad_(grad)

    def _crop(self, left_x, right_y):
        # legacy, in case input are not always divisible by 2
        start = (left_x.shape[2] - right_y.shape[2]) // 2
        end = right_y.shape[2]+start
        return left_x[:, :, start:end, start:end]

    def _make_contracting_path(self):
        feature = self.init_features
        res = []
        for i in range(-1, self.n_block - 1):
            if i == -1:
                block = self.contract_block_fn(self.in_channels,
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
            ri = self.n_block - i - 1
            block = self.expand_block_fn(feature * 2 ** (ri+1),
                                         feature * 2 ** (ri))

            res.append(block)
        self.expanding_path = nn.ModuleList(res)

    def _make_bottleneck(self):
        b = self.bottleneck_block_fn(
            self.init_features * 2 ** (self.n_block-1),
            self.init_features * 2 ** (self.n_block))
        self.bottleneck = b

    def _make_last(self):
        self.last = nn.Conv2d(self.init_features, self.out_channels,
                              kernel_size=1)
        self.out = nn.Softmax(dim=1)

    def _make_layer(self):
        self._make_contracting_path()
        self._make_bottleneck()
        self._make_expanding_path()
        self._make_last()

    def forward(self, x):
        xt = []
        for block in self.contracting_path:
            x, tmp = block(x)
            xt.append(tmp)

        y = self.bottleneck(x)

        for i, block in enumerate(self.expanding_path):
            y = block(y, xt[-(i+1)])

        # for i in range(8):  # for visual
        #     show(*y[0][i*8:i*8+8].cpu(), width=4)
        return self.out(self.last(y))

    def display_grad_per_block(self, model=None, name=''):
        if model is None:
            model = self
        if len(list(model.children())) > 0:
            for n, module in model.named_children():
                self.display_grad_per_block(model=module, name=f'{name} - {n}')
        else:
            msg = ''
            for n, p in model.named_parameters():
                msg += f'| {n}: {p.requires_grad}'
            if msg != '':
                print(f'{name} - {msg}')

    def initialize_weights(self, model=None):
        if model is None:
            model = self
        if len(list(model.children())) > 0:
            for n, module in model.named_children():
                self.initialize_weights(model=module)
        else:
            if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear):
                nn.init.kaiming_normal_(model.weight)
                if model.bias is not None:
                    model.bias.data.zero_()
            elif isinstance(model, nn.BatchNorm2d):
                model.weight.data.fill_(1)
                model.bias.data.zero_()


class ClassicalUnet(Unet):
    def __init__(self, **kwargs):
        upblock_fn = partial(UpConvBlockConv, residual=False, bn=False)
        downblock_fn = partial(DownConvBlockMax, residual=False, bn=True)
        block_fn = partial(ConvBlock, residual=False, bn=True)
        super(ClassicalUnet, self).__init__(downblock_fn, upblock_fn,
                                            block_fn, **kwargs)


class ResidualUnet(Unet):
    def __init__(self, **kwargs):
        upblock_fn = partial(UpConvBlockConv, residual=True, bn=True)
        downblock_fn = partial(DownConvBlockMax, residual=True, bn=True)
        block_fn = partial(ConvBlock, residual=True, bn=True)
        super(ResidualUnet, self).__init__(downblock_fn, upblock_fn,
                                           block_fn, **kwargs)
