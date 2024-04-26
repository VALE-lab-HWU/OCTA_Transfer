import torch
import torch.nn as nn


class UpConvGeneric(nn.Module):
    def __init__(self, up_block, conv_block):
        super().__init__()
        self.up = up_block
        self.conv = conv_block

    def forward(self, x, skip_x):
        return self.conv(torch.cat([self.up(x), skip_x], dim=1))
