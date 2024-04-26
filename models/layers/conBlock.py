import torch.nn as nn


# it's just a generic sequential in fact
# but oh well
# it's easier to understand this way
class ConConvGeneric(nn.Module):
    def __init__(self, resample_block, conv_block):
        super().__init__()
        self.resample = resample_block
        self.conv = conv_block

    def forward(self, x):
        return self.conv(self.resample(x))
