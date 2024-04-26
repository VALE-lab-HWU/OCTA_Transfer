from torch import nn
from torchvision.models import inception_v3


# weight IMAGENET1K_V1
def get_inception_v3_flim(in_channels=1, out_channels=2, pretrained=False):
    if pretrained:
        weights = 'IMAGENET1K_V1'
    else:
        weights = None
    md = inception_v3(weights=weights)
    md.Conv2d_1a_3x3.conv = nn.Conv2d(
        in_channels, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    md.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=out_channels, bias=True))
    return md
