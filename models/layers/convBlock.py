import torch
import torch.nn as nn
from .upBlock import UpConvGeneric
from .downBlock import DownConvGeneric
from .conBlock import ConConvGeneric
from .attentionBlock_mine import UpConvBlockAttention, ConvUnextAG, \
    CrossChannelAttention, AttentionGate


class depthWiseSepConv(nn.Sequential):
    def __init__(self, in_channels, out_channelsz,
                 kernel_size=3, padding=1, stride=1, **kwargs):
        super(depthWiseSepConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=kernel_size, padding=padding,
                               stride=stride, groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channelsz, kernel_size=1)


class ConvBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels,
                 resample=None, stride=1, kernel_size=3, padding=1,
                 bn=True, residual=True, block=nn.Conv2d, **kwargs):
        super(ConvBlockBase, self).__init__()
        self.residual = residual
        seq_block = []
        bias = True  # not bn
        seq_block.append(block(in_channels, out_channels,
                               kernel_size=kernel_size,
                               padding=padding, stride=stride,
                               bias=bias))
        if bn:
            seq_block.append(nn.BatchNorm2d(out_channels))
        seq_block.append(nn.ReLU(inplace=True))
        seq_block.append(block(out_channels, out_channels,
                               kernel_size=kernel_size, padding=padding,
                               bias=bias))
        if bn:
            seq_block.append(nn.BatchNorm2d(out_channels))
        self.conv = nn.Sequential(*seq_block)
        self.relu2 = nn.ReLU(inplace=True)
        if residual:
            self.resample = resample

    def forward(self, x):
        identity = x
        out = self.conv(x)

        if self.residual:
            if self.resample is not None:
                identity = self.resample(x)
            out = out + identity

        out = self.relu2(out)
        return out


class ConvBlockOneBase(nn.Module):
    def __init__(self, in_channels, out_channels,
                 resample=None, stride=1, kernel_size=3, padding=1,
                 bn=True, residual=True, **kwargs):
        super(ConvBlockOneBase, self).__init__()
        self.residual = residual
        seq_block = []
        seq_block.append(nn.Conv2d(in_channels, out_channels,
                                   kernel_size=kernel_size,
                                   padding=padding, stride=stride))
        if bn:
            seq_block.append(nn.BatchNorm2d(out_channels))
        self.conv = nn.Sequential(*seq_block)
        self.relu = nn.ReLU(inplace=True)
        if residual:
            self.resample = resample

    def forward(self, x):
        identity = x
        out = self.conv(x)

        if self.residual:
            if self.resample is not None:
                identity = self.resample(x)
            out = out + identity

        out = self.relu(out)
        return out


class ConvBlockNBase(nn.Module):
    def __init__(self, in_channels, out_channels, n_block=2,
                 resample=None, stride=1, kernel_size=3, padding=1,
                 bn=True, residual=True, block=ConvBlockOneBase, **kwargs):
        super(ConvBlockNBase, self).__init__()
        self.residual = residual
        seq_block = []
        seq_block.append(block(
            in_channels, out_channels, residual=False, kernel_size=kernel_size,
            stride=stride, padding=padding, bn=bn))
        for i in range(1, n_block):
            seq_block.append(block(
                out_channels, out_channels, residual=False, stride=stride,
                kernel_size=kernel_size, padding=padding, bn=bn))
        self.conv = nn.Sequential(*seq_block)
        self.relu = nn.ReLU(inplace=True)
        if residual:
            self.resample = resample

    def forward(self, x):
        identity = x
        out = self.conv(x)

        if self.residual:
            if self.resample is not None:
                identity = self.resample(x)
            out = out + identity
            out = self.relu(out)
        return out


class ConvBlock(ConvBlockBase):
    def __init__(self, in_channels, out_channels, stride=1,
                 **kwargs):
        if out_channels != in_channels:
            sample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            sample = None
        super(ConvBlock, self).__init__(in_channels, out_channels,
                                        resample=sample, stride=stride,
                                        **kwargs)


class ConvBlockOne(ConvBlockOneBase):
    def __init__(self, in_channels, out_channels, stride=1,
                 **kwargs):
        if out_channels != in_channels:
            sample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            sample = None
        super(ConvBlockOne, self).__init__(in_channels, out_channels,
                                           resample=sample, stride=stride,
                                           **kwargs)


class ConvBlockN(ConvBlockNBase):
    def __init__(self, in_channels, out_channels, stride=1,
                 **kwargs):
        if out_channels != in_channels:
            sample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            sample = None
        super(ConvBlockN, self).__init__(in_channels, out_channels,
                                         resample=sample, stride=stride,
                                         **kwargs)


###
# # upblock
###
class UpConvBlockConv(UpConvGeneric):
    def __init__(self, in_channels, out_channels, mid_channels=None, sc=2,
                 **kwargs):
        if mid_channels is None:
            mid_channels = in_channels
        conv = ConvBlock(in_channels, out_channels, **kwargs)
        up = nn.ConvTranspose2d(mid_channels, out_channels,
                                kernel_size=sc, stride=sc)
        super().__init__(up, conv)


class UpConvBlockSample(UpConvGeneric):
    def __init__(self, in_channels, out_channels, mid_channels=None, sc=2,
                 **kwargs):
        conv = ConvBlock(in_channels, out_channels, **kwargs)
        up = nn.Upsample(scale_factor=sc, mode='bilinear')
        super().__init__(up, conv)


# upblock with attention
class UpConvBlockAG(UpConvBlockAttention):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 skip_channels=None, sc=2, ag_k=1, ag_s=1,
                 **kwargs):
        if skip_channels is None:
            skip_channels = out_channels
        if mid_channels is None:
            mid_channels = in_channels
        up = nn.ConvTranspose2d(mid_channels, out_channels,
                                kernel_size=sc, stride=sc)
        conv = ConvBlock(in_channels, out_channels, **kwargs)
        ag = AttentionGate(skip_channels, out_channels, skip_channels//2,
                           kernel=ag_k, scale=ag_s)
        super().__init__(up, conv, ag)


class UpConvBlockAGPaper(UpConvBlockAttention):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 skip_channels=None, sc=2, ag_k=2, ag_s=1,
                 **kwargs):
        if skip_channels is None:
            skip_channels = out_channels
        if mid_channels is None:
            mid_channels = in_channels
        up = nn.ConvTranspose2d(mid_channels, out_channels,
                                kernel_size=sc, stride=sc)
        conv = ConvBlock(in_channels, out_channels, **kwargs)
        ag = AttentionGate(skip_channels, in_channels, skip_channels//2,
                           kernel=ag_k, scale=ag_s)
        super().__init__(up, conv, ag)

    def forward(self, x, skip_x):
        skip_x = self.ag(x, skip_x)
        x = torch.cat([self.up(x), skip_x], dim=1)
        if self.conv1 is not None:
            x = self.conv1(x)
        return self.conv(x)


class UpConvBlockCCA(UpConvBlockAttention):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 skip_channels=None, sc=2, ag_k=1, ag_s=1,
                 **kwargs):
        if skip_channels is None:
            skip_channels = out_channels
        if mid_channels is None:
            mid_channels = in_channels
        up = nn.ConvTranspose2d(mid_channels, out_channels,
                                kernel_size=sc, stride=sc)
        conv = ConvBlock(in_channels, out_channels, **kwargs)
        ag = CrossChannelAttention(skip_channels, out_channels)
        super().__init__(up, conv, ag)


class UpConvBlockNextAG(UpConvBlockAttention):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 skip_channels=None, sc=2, ag_k=1, ag_s=1,
                 **kwargs):
        if skip_channels is None:
            skip_channels = out_channels
        if mid_channels is None:
            mid_channels = in_channels
        up = nn.ConvTranspose2d(mid_channels, out_channels,
                                kernel_size=sc, stride=sc)
        conv = ConvBlock(in_channels, out_channels, **kwargs)
        ag = ConvUnextAG(skip_channels, out_channels)
        super().__init__(up, conv, ag)


###
# # downblock
###
class DownConvBlockMax(DownConvGeneric):
    def __init__(self, in_channels, out_channels, mid_channels=None,  sc=2,
                 **kwargs):
        conv = ConvBlock(in_channels, out_channels, **kwargs)
        down = nn.MaxPool2d(kernel_size=sc, stride=sc)
        super().__init__(down, conv)


class DownConvBlockOneMax(DownConvGeneric):
    def __init__(self, in_channels, out_channels, mid_channels=None,  sc=2,
                 **kwargs):
        conv = ConvBlockOne(in_channels, out_channels, **kwargs)
        down = nn.MaxPool2d(kernel_size=sc, stride=sc)
        super().__init__(down, conv)


class DownConvBlockNMax(DownConvGeneric):
    def __init__(self, in_channels, out_channels, mid_channels=None,  sc=2,
                 **kwargs):
        conv = ConvBlockN(in_channels, out_channels, **kwargs)
        down = nn.MaxPool2d(kernel_size=sc, stride=sc)
        super().__init__(down, conv)


###
# # connections
###
class ConConvBlockUp(ConConvGeneric):
    def __init__(self, in_channels, out_channels, sc=2, **kwargs):
        conv = ConvBlock(in_channels, out_channels, **kwargs)
        up = nn.Upsample(scale_factor=sc, mode='bilinear')
        super().__init__(up, conv)


class ConConvBlockDown(ConConvGeneric):
    def __init__(self, in_channels, out_channels, sc=2, **kwargs):
        conv = ConvBlock(in_channels, out_channels, **kwargs)
        down = nn.MaxPool2d(kernel_size=sc, stride=sc)
        super().__init__(down, conv)


class ConConvBlockOneUp(ConConvGeneric):
    def __init__(self, in_channels, out_channels, sc=2, **kwargs):
        conv = ConvBlockOne(in_channels, out_channels, **kwargs)
        up = nn.Upsample(scale_factor=sc, mode='bilinear')
        super().__init__(up, conv)


class ConConvBlockOneDown(ConConvGeneric):
    def __init__(self, in_channels, out_channels, sc=2, **kwargs):
        conv = ConvBlockOne(in_channels, out_channels, **kwargs)
        down = nn.MaxPool2d(kernel_size=sc, stride=sc)
        super().__init__(down, conv)
