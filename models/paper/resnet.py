from torch import nn
from torchvision.models import resnet50


# weight IMAGENET1K_V2
def get_resnet_50_flim(in_channels=1, out_channels=2, pretrained=False):
    if pretrained:
        weights = 'IMAGENET1K_V2'
    else:
        weights = None
    md = resnet50(weights=weights)
    md.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2),
                         padding=(3, 3), bias=False)
    md.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=out_channels, bias=True))
    return md
