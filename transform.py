import numpy as np
import torch
from torchvision import transforms
LOG = False


# rewrite in python accounting torch of the blend
# function within the core library of PIL
# originally in C, here
# https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Blend.c
#
# im1 is degenerate for enhancement function
# im2 is the image to enhance
def blend(im1, im2, alpha, clip=False, clip_l=None, clip_h=None):
    if alpha == 0:
        return im1.detach().clone()
    elif alpha == 1:
        return im2.detach().clone()

    if alpha >= 0 and alpha <= 1:
        # interpolate
        out = im1 + alpha * (im2 - im1)
    else:
        # extrapolate
        # has to be within clip value
        out = im1 + alpha * (im2 - im1)
        if clip:
            out = out.clip(clip_l, clip_h)
    return out


class Crop(object):
    def __init__(self, crop_prob, size):
        self.crop_prob = crop_prob
        self.size = size

    def __call__(self, sample):
        res = sample
        if np.random.rand() < self.crop_prob:
            if LOG:
                print('crop true')
            if len(self.size) == 2:
                size = np.random.randint(*self.size)
            else:
                size = self.size[0]
            x = np.random.randint(0, sample[0].shape[-1]-size)
            y = np.random.randint(0, sample[0].shape[-2]-size)
            res = [transforms.functional.crop(i, x, y, size, size)
                   for i in sample]
        else:
            if LOG:
                print('crop false')
        return res


class Erase(transforms.RandomErasing):
    def __init__(self, erase_prob, erase_scale, erase_ratio, erase_value,
                 only_x=True):
        super(Erase, self).__init__(1, scale=erase_scale,
                                    ratio=erase_ratio, value=erase_value)
        self.only_x = only_x
        self.erase_prob = erase_prob

    def __call__(self, sample):
        res = sample
        if np.random.rand() < self.erase_prob:
            if LOG:
                print('erase true')
            if self.only_x:
                x = super(Erase, self).__call__(sample[0])
                res = [x, *sample[1:]]
            else:
                res = [super(Erase, self).__call__(i) for i in res]
        else:
            if LOG:
                print('erase false')
        return res


class Invert(object):
    def __init__(self, invert_prob, minv=0., maxv=1., only_x=True):
        self.invert_prob = invert_prob
        self.min = minv
        self.max = maxv
        self.only_x = only_x

    def __call__(self, sample):
        res = sample
        if np.random.rand() < self.invert_prob:
            if LOG:
                print('invert true')
            if self.only_x:
                x = transforms.functional.invert(
                    sample[0]) + self.max + self.min
                res = [x, *sample[1:]]
            else:
                res = [transforms.functional.invert(i) + self.max + self.min
                       for i in sample]
        else:
            if LOG:
                print('invert false')
        return res


class Brightness(object):
    def __init__(self, factor_min, factor_max, clip=False, only_x=True,
                 clip_l=None, clip_h=None, min_0=True):
        self.factor_min = factor_min
        self.factor_max = factor_max
        self.only_x = only_x
        self.clip_l = clip_l
        self.clip_h = clip_h
        self.clip = clip
        self.min_0 = min_0

    def __call__(self, sample):
        factor = np.random.uniform(self.factor_min, self.factor_max)
        if LOG:
            print('bright factor', factor)
        if self.only_x:
            if self.min_0:
                min_0 = 0
            else:
                min_0 = sample[0].min()
            degenerate = torch.full(size=sample[0].shape, fill_value=min_0)
            x = blend(degenerate, sample[0], factor, clip=self.clip,
                      clip_l=self.clip_l, clip_h=self.clip_h)
            res = [x, *sample[1:]]
        else:
            if self.min_0:
                res = [blend(torch.zeros(size=i.shape), i, factor,
                             clip=self.clip, clip_l=self.clip_l,
                             clip_h=self.clip_h) for i in sample]
            else:
                res = [blend(torch.full(i.shape, i.min()), i, factor,
                             clip=self.clip, clip_l=self.clip_l,
                             clip_h=self.clip_h) for i in sample]
        return res


class Contrast(object):
    def __init__(self, factor_min, factor_max, clip=False, only_x=True,
                 clip_l=None, clip_h=None):
        self.factor_min = factor_min
        self.factor_max = factor_max
        self.only_x = only_x
        self.clip_l = clip_l
        self.clip_h = clip_h
        self.clip = clip

    def __call__(self, sample):
        factor = np.random.uniform(self.factor_min, self.factor_max)
        if LOG:
            print('contrast factor', factor)
        if self.only_x:
            degenerate = torch.full(size=sample[0].shape,
                                    fill_value=sample[0].mean())
            x = blend(degenerate, sample[0], factor, clip=self.clip,
                      clip_l=self.clip_l, clip_h=self.clip_h)
            res = [x, *sample[1:]]
        else:
            res = [blend(torch.full(size=i.shape, fill_value=sample[0].mean()),
                         i, factor, clip=self.clip,
                         clip_l=self.clip_l, clip_h=self.clip_h)
                   for i in sample]
        return res


class Flip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        res = sample
        if np.random.rand() < self.flip_prob:
            if np.random.rand() > 0.5:  # hflip
                if LOG:
                    print('flip horizontal')
                res = [transforms.functional.hflip(i).clone() for i in res]
            else:  # vflip
                if LOG:
                    print('flip vertical')
                res = [transforms.functional.vflip(i).clone() for i in res]
        else:
            if LOG:
                print('flip false')
        return res


class Rotate(object):
    def __init__(self, angle, fill=False):
        self.angle = angle
        self.fill = fill

    def __call__(self, sample):
        angle = np.random.uniform(-self.angle, self.angle)
        if LOG:
            print('rotate angle', angle)
        if self.fill:
            res = [transforms.functional.rotate(
                i, angle, fill=float(i[0][0][0])) for i in sample]
        else:
            res = [transforms.functional.rotate(i, angle) for i in sample]
        return res


class Blur(object):
    def __init__(self, kernel, sigma_min, sigma_max, only_x=True):
        self.kernel = kernel
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.only_x = only_x

    def __call__(self, sample):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        if LOG:
            print('blur sigma', sigma)
        if self.only_x:
            x = transforms.functional.gaussian_blur(sample[0], self.kernel,
                                                    sigma)
            res = [x, *sample[1:]]
        else:
            res = [transforms.functional.gaussian_blur(i, self.kernel, sigma)
                   for i in sample]
        return res


def get_transforms(angle=None, flip_prob=None, kernel=None, sigma=None,
                   bright=None, contrast=None, invert_prob=None, fill_r=False,
                   erase_prob=None, erase_scale=None, erase_ratio=None,
                   erase_value=None, crop_prob=None, crop_size=None,
                   clip=False, clip_l=None, clip_h=None, only_x=True,
                   min_0=True):
    transform_list = []
    if invert_prob is not None:
        transform_list.append(Invert(invert_prob, minv=clip_l, maxv=clip_h,
                                     only_x=only_x))
    if bright is not None:
        transform_list.append(Brightness(
            bright[0], bright[1], only_x=only_x, clip=clip, clip_l=clip_l,
            clip_h=clip_h, min_0=min_0))
    if contrast is not None:
        transform_list.append(Contrast(
            contrast[0], contrast[1], only_x=only_x, clip=clip, clip_l=clip_l,
            clip_h=clip_h))
    if kernel is not None and sigma is not None:
        transform_list.append(Blur(kernel, sigma[0], sigma[1], only_x=only_x))
    if angle is not None:
        transform_list.append(Rotate(angle, fill=fill_r))
    if flip_prob is not None:
        transform_list.append(Flip(flip_prob))
    if crop_prob is not None:
        transform_list.append(Crop(crop_prob, crop_size))
    if erase_prob is not None:
        transform_list.append(Erase(erase_prob, erase_scale,
                                    erase_ratio, erase_value))
    return transforms.Compose(transform_list)


class FixedCrop(object):
    def __init__(self, size):
        self.size = size

    # image CxHxW
    def __call__(self, sample):
        res = sample
        # shape = [i.shape for i in res]
        res = [i.unfold(2, self.size, self.size)
               .unfold(1, self.size, self.size)
               .reshape(-1, i.shape[0], self.size, self.size)
               for i in res]
        return res

    def reverse(sample, shape):
        res = sample
        res = [v.permute(0, 1, 4, 2, 3).contiguous().view(shape[i])
               for i, v in enumerate(res)]
        return res


class Pad(object):
    def __init__(self, padding, value):
        self.padding = padding
        self.value = value

    def __call__(self, sample):
        res = sample
        res = [torch.nn.functional.pad(i, self.padding, value=self.value)
               for i in res]
        return res


class PadToSize(object):
    def __init__(self, size, value):
        self.size = size  # C x H x W
        self.value = value

    def __call__(self, sample):
        res = sample
        res = [torch.nn.functional.pad(
            i, tuple(k for j, s in enumerate(i.shape)
                     for k in [self.size[j]-s, 0])[::-1],
            value=self.value)
               for i in res]
        return res


def get_test_transforms(pad_size=None, padding=None, pad_value=None,
                        crop_size=None):
    transform_list = []
    if padding is not None:
        transform_list.append(Pad(padding, pad_value))
    if pad_size is not None:
        transform_list.append(PadToSize(pad_size, pad_value))
    if crop_size is not None:
        transform_list.append(FixedCrop(crop_size))
    return transforms.Compose(transform_list)
