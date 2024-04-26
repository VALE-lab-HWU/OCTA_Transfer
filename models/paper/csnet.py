"""
Channel and Spatial CSNet Network (CS-Net).
Code from https://github.com/iMED-Lab/CS-Net/blob/master/model/csnet.py
Licensed under MIT
Copyright (c) 2020 ineedzx
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


def downsample():
    return nn.MaxPool2d(kernel_size=2, stride=2)


def deconv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels,
                              kernel_size=2, stride=2)


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = out + residual  # LINE CHANGED to fix runtime issue
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//8,
                      kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(inplace=True)
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//8,
                      kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(inplace=True)
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = (torch.max(affinity, -1, keepdim=True)[0]
                        .expand_as(affinity)) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class AffinityAttention(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttention, self).__init__()
        self.sab = SpatialAttentionBlock(in_channels)
        self.cab = ChannelAttentionBlock(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        cab = self.cab(x)
        out = sab + cab
        return out


class CSNet(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(CSNet, self).__init__()
        self.contracting_path_0 = ResEncoder(in_channels, 64)
        self.contracting_path_1 = ResEncoder(64, 128)
        self.contracting_path_2 = ResEncoder(128, 256)
        self.contracting_path_3 = ResEncoder(256, 512)
        self.bottleneck = ResEncoder(512, 1024)
        self.downsample = downsample()
        self.affinity_attention = AffinityAttention(1024)
        # self.a = nn.Conv2d(1024 * 2, 1024, kernel_size=1)
        self.expanding_path_0_conv = Decoder(1024, 512)
        self.expanding_path_1_conv = Decoder(512, 256)
        self.expanding_path_2_conv = Decoder(256, 128)
        self.expanding_path_3_conv = Decoder(128, 64)
        self.expanding_path_0_up = deconv(1024, 512)
        self.expanding_path_1_up = deconv(512, 256)
        self.expanding_path_2_up = deconv(256, 128)
        self.expanding_path_3_up = deconv(128, 64)
        self.last = nn.Conv2d(64, out_channels, kernel_size=1)
        initialize_weights(self)

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

    def forward(self, x):
        enc_input = self.contracting_path_0(x)
        down1 = self.downsample(enc_input)

        enc1 = self.contracting_path_1(down1)
        down2 = self.downsample(enc1)

        enc2 = self.contracting_path_2(down2)
        down3 = self.downsample(enc2)

        enc3 = self.contracting_path_3(down3)
        down4 = self.downsample(enc3)

        input_feature = self.bottleneck(down4)

        # Do Attenttion operations here
        attention = self.affinity_attention(input_feature)

        # attention_fuse = self.attention_fuse(torch.cat(
        #     (input_feature, attention), dim=1))
        attention_fuse = input_feature + attention

        # Do decoder operations here
        up4 = self.expanding_path_0_up(attention_fuse)
        up4 = torch.cat((enc3, up4), dim=1)
        dec4 = self.expanding_path_0_conv(up4)

        up3 = self.expanding_path_1_up(dec4)
        up3 = torch.cat((enc2, up3), dim=1)
        dec3 = self.expanding_path_1_conv(up3)

        up2 = self.expanding_path_2_up(dec3)
        up2 = torch.cat((enc1, up2), dim=1)
        dec2 = self.expanding_path_2_conv(up2)

        up1 = self.expanding_path_3_up(dec2)
        up1 = torch.cat((enc_input, up1), dim=1)
        dec1 = self.expanding_path_3_conv(up1)

        final = self.last(dec1)
        # final = F.sigmoid(final)
        final = F.softmax(final, dim=1)
        return final
