import torch
import torch.nn as nn
import torch.nn.functional as F
from .upBlock import UpConvGeneric


class SpatialAttentionBlock(nn.Module):
    """PAM"""
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        # add the value to the .parameters() of the module
        self.alpha = nn.Parameter(torch.zeros(1))
        self.b_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.c_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.d_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, a):
        nb, chan, height, width = a.size()
        nhw = height * width
        b = self.b_conv(a)
        c = self.c_conv(a)
        b_prime = b.view(nb, -1, nhw)
        c_prime = c.view(nb, -1, nhw)
        k = torch.matmul(b_prime.transpose(1, 2), c_prime)
        s = self.softmax(k)
        d = self.d_conv(a)
        d_prime = d.view(nb, -1, nhw)
        kl = torch.matmul(d_prime, s.transpose(1, 2))
        kl_prime = kl.view(nb, chan, height, width)
        e = self.alpha * kl_prime + a
        return e


class ChannelAttentionBlock(nn.Module):
    """CAM"""
    def __init__(self):
        super(ChannelAttentionBlock, self).__init__()
        # add the value to the .parameters() of the module
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, a):
        nb, chan, height, width = a.size()
        a_prime = a.view(nb, chan, -1)
        b = torch.matmul(a_prime, a_prime.transpose(1, 2))
        x = self.softmax(b)
        c = torch.matmul(x, a_prime)
        c_prime = c.view(nb, chan, height, width)
        e = self.beta * c_prime + a
        return e


# class UAttentionBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, block, **kwargs):
#         super(UAttentionBlock, self).__init__()
#         self.key = block(in_channels=in_channels,
#                          out_channels=out_channels//8,
#                          **kwargs)
#         self.value = block(in_channels=in_channels,
#                            out_channels=out_channels//8,
#                            **kwargs)
#         self.softmax = nn.Softmax(dim=-1)
#         self.alpha = nn.Parameter(torch.zeros(1))

#     def forward(self, x, x2):
#         nb, chan, height, width = x.size()
#         nhw = height * width
#         f = x2.view(nb, -1, nhw).transpose(1, 2)
#         g = self.key(x).view(nb, -1, nhw)
#         h = self.value(x).view(nb, -1, nhw)
#         beta = self.softmax(torch.bmm(g, f))
#         out = self.alpha * torch.bmm(beta.transpose(1, 2), h)
#         out = out.view(nb, chan, height, width).contiguous()
#         return out + x


class AttentionGate(nn.Module):
    def __init__(self, in_skip_channels, in_gate_channels, mid_channels,
                 kernel=2, scale=1):
        super().__init__()
        self.gate_conv = nn.Conv2d(in_gate_channels,  mid_channels,
                                   kernel_size=1, stride=1)
        self.skip_conv = nn.Conv2d(in_skip_channels,  mid_channels,
                                   kernel_size=kernel, stride=kernel,
                                   bias=False)
        self.upsample_skip = nn.Upsample(scale_factor=scale)
        self.relu = nn.ReLU(inplace=True)
        self.attention_conv = nn.Conv2d(mid_channels, 1,
                                        kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.upsample_attn = nn.Upsample(scale_factor=kernel)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_skip_channels,
                      out_channels=in_skip_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_skip_channels),
        )

    def forward(self, x, skip_x):
        x = self.gate_conv(x)
        skip_x_up = self.upsample_skip(self.skip_conv(skip_x))
        attn = self.relu(x + skip_x_up)
        attn = self.upsample_attn(self.sigmoid(self.attention_conv(attn)))
        skip_x = attn.expand_as(skip_x) * skip_x
        return self.out_conv(skip_x)


class CrossChannelAttention(nn.Module):
    def __init__(self, skip_channels, gate_channels):
        super().__init__()
        self.linear_skip = nn.Linear(skip_channels, skip_channels)
        self.linear_gate = nn.Linear(gate_channels, skip_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip_x):
        # stride equal to kernel size default
        # not using layers, cause it is dependant on the size, which we want
        # to be invariant to
        pooled_x = F.avg_pool2d(x, kernel_size=x.shape[-2:]).view(len(x), -1)
        pooled_skip_x = F.avg_pool2d(x, kernel_size=x.shape[-2:]).view(len(x), -1)
        pooled_x = self.linear_gate(pooled_x)
        pooled_skip_x = self.linear_skip(pooled_skip_x)
        attn = F.sigmoid((pooled_x + pooled_skip_x) / 2.0)
        attn = attn.view(*attn.shape, 1, 1).expand_as(skip_x)
        return self.relu(attn * x)


class ConvUnextAG(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gate = nn.Linear(in_channels, 3 * in_channels)
        self.linear_g1 = nn.Linear(in_channels, in_channels)
        self.linear_skip = nn.Linear(in_channels, in_channels)

    def forward(self, x, skip_x):
        x = x.permute(0, 2, 3, 1)
        skip_x = skip_x.permute(0, 2, 3, 1)
        x1, x2, x3 = self.gate(x).reshape(*x.shape, 3).permute(4, 0, 1, 2, 3)
        skip_x = F.sigmoid(self.linear_g1(x1 + skip_x)) * skip_x \
            + F.sigmoid(x2) * F.tanh(x3)
        skip_x = self.linear_skip(skip_x)
        return skip_x.permute(0, 3, 1, 2)


class UpConvBlockAttention(UpConvGeneric):
    def __init__(self, up, conv, ag, conv1=None,
                 **kwargs):
        super().__init__(up, conv)
        self.ag = ag
        self.conv1 = conv1

    def forward(self, x, skip_x):
        x = self.up(x)
        skip_x = self.ag(x, skip_x)
        x = torch.cat([x, skip_x], dim=1)
        if self.conv1 is not None:
            x = self.conv1(x)
        return self.conv(x)
