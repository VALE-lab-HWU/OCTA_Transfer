import torch.nn as nn


class DownConvGeneric(nn.Module):
    def __init__(self, down_block, conv_block):
        super().__init__()
        self.down = down_block
        self.conv = conv_block

    def forward(self, x):
        res = self.conv(x)
        return self.down(res), res
