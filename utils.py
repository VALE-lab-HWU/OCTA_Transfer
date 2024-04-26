import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil


def log(msg, log_lv, log=0):
    if log_lv >= log:
        print(msg)


def t_n(tens, b=False):
    arange = torch.arange(tens.ndim)
    if b:
        return tens.permute(0, *arange[-2:], *arange[1:-2])
    else:
        return tens.permute(*arange[-2:], *arange[:-2])


def n_t(tens, b=False):
    arange = torch.arange(tens.ndim)
    if b:
        return tens.permute(0, *arange[3:], *arange[1:3])
    else:
        return tens.permute(*arange[2:], *arange[:2])


def store_results(title='dl', name='result', **res):
    pd.to_pickle(res, f'./results/{title}/{name}.pkl')


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def show(*x, title=[], width=None, block=True, **kwargs):
    ti = len(title) > 0
    if len(x) == 1:
        plt.imshow(x[0], **kwargs)
        plt.axis('off')
    else:
        if width is None:
            width = len(x)
            height = 1
        else:
            height = ceil(len(x) / width)
        fig, axs = plt.subplots(height, width)
        for i in range(height):
            for j in range(width):
                if j+i*width >= len(x):
                    break
                if height == 1:
                    axs[j].imshow(x[j+i*width], **kwargs)
                    axs[j].set_axis_off()
                    if ti:
                        axs[j].set_title(title[j+i*width])
                else:
                    axs[i][j].imshow(x[j+i*width], **kwargs)
                    axs[i][j].set_axis_off()
                    if ti:
                        axs[i][j].set_title(title[j+i*width])
    plt.show(block=block)


def seg_res_img(y_p, y_t):
    res = torch.zeros((3, *y_p.shape))
    idx_common = y_p == y_t
    idx_true_1 = y_t == 1
    res[1][torch.logical_and(idx_common, idx_true_1)] = 255
    res[0][torch.logical_and(~idx_common, ~idx_true_1)] = 255
    res[2][torch.logical_and(~idx_common, idx_true_1)] = 255
    return res.int()
