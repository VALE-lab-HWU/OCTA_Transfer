import torch
import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(self, classes=2, one_hot=True, ignore=0, smooth=1e-5):
        super(CustomLoss, self).__init__()
        self.classes = classes
        self.one_hot = one_hot
        self.ignore = ignore
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        dsc = CustomLoss.multi_loss(y_pred.cpu(), y_true.cpu(),
                                    classes=self.classes,
                                    one_hot=self.one_hot, ignore=self.ignore)
        return 1. - dsc

    # y_pred and y_true should be of shape (-1, classes)
    # cause otherwise can't do different size..
    def multi_loss(y_pred, y_true, classes=2,
                   one_hot=True, ignore=None, smooth=1e-5,
                   fn='dice'):
        if fn == 'dice':
            fn = CustomLoss.dice
        elif fn == 'iou':
            fn = CustomLoss.iou
        else:
            raise ValueError('wrong loss')
        dscs = torch.tensor([0.])
        if one_hot:
            for i in range(classes):
                if not i == ignore:
                    dscs += (fn(
                        y_pred[:, i], y_true == i, smooth))
        else:
            for i in range(classes):
                if not i == ignore:
                    dscs += (fn(y_pred == i,
                                y_true == i,
                                smooth))
        dsc = dscs / (classes if ignore is None else classes-1)
        return dsc

    def dice(y_pred, y_true, smooth):
        y_pred = y_pred[:].contiguous().view(-1)
        y_true = y_true[:].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2 * intersection + smooth) / (
            y_pred.sum() + y_true.sum() + smooth
        )
        return dsc

    def iou(y_pred, y_true, smooth):
        y_pred = y_pred[:].contiguous().view(-1)
        y_true = y_true[:].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (intersection + smooth) / (
            y_pred.sum() + y_true.sum() - intersection + smooth
        )
        return dsc
