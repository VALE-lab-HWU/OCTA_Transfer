import os
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from functools import partial


from arg import parse_args
from DatasetHandler import DatasetHandler
from collate_fn import my_collate

from utils import log, store_results, mkdir
from ml_helper import compare_class
from transform import get_transforms, get_test_transforms
import dl_helper
from loss.seg_loss import CustomLoss
from models.unet import ClassicalUnet, ResidualUnet
from models.csnet import ClassicalCSnet, ResidualCSnet
from models.unetpp import ClassicalUnetPlusPlus as ClassicalUnetPP, \
    ResidualUnetPlusPlus as ResidualUnetPP
from models.unet3p import ClassicalUnet3Plus, ResidualUnet3Plus, PaperUnet3Plus
from models.uctransnet import PaperUCTransNet, ClassicalUCTransNet, \
    ResidualUCTransNet

from models.paper.csnet import CSNet
from models.paper.UNet import UNet
from models.paper.UNet_2Plus import UNet_2Plus
from models.paper.UNet_3Plus import UNet_3Plus
from models.paper.UCTransNet import UCT


MD_DICT_MINE = {'unet_c': ClassicalUnet, 'unet_r': ResidualUnet,
                'csnet_c': ClassicalCSnet, 'csnet_r': ResidualCSnet,
                'unet++_c': ClassicalUnetPP, 'unetpp_c': ClassicalUnetPP,
                'unet++_r': ResidualUnetPP, 'unetpp_r': ResidualUnetPP,
                'unet3+_c': ClassicalUnet3Plus, 'unet3p_c': ClassicalUnet3Plus,
                'unet3+_r': ResidualUnet3Plus, 'unet3p_r': ResidualUnet3Plus,
                'unet3+_p': PaperUnet3Plus, 'unet3p_p': PaperUnet3Plus,
                'uctransnet_p': PaperUCTransNet,
                'uctransnet_c': ClassicalUCTransNet,
                'uctransnet_r': ResidualUCTransNet}

MD_DICT_PAPER = {'p_csnet': CSNet, 'p_unet': UNet,
                 'p_unet++': UNet_2Plus, 'p_unetpp': UNet_2Plus,
                 'p_unet3+': UNet_3Plus, 'p_unet3p': UNet_3Plus,
                 'p_uctransnet': UCT}

MD_DICT = {**MD_DICT_MINE, **MD_DICT_PAPER}


def test_model_fn(model, ts_dl, title, name, args, device, val=False):
    print('---------')
    print(f'Testing {title} {name}')
    y_pred, y_true = dl_helper.test(
        ts_dl, model, grad_acc=args.grad_acc, crop=args.tf_crop_test,
        device=device, log=args.log, my_ds=not val)
    y_pred = torch.argmax(y_pred, dim=1)
    y_pred = y_pred[:].contiguous().view(-1).cpu()
    y_true = y_true[:].contiguous().view(-1).cpu()
    dice = CustomLoss.multi_loss(y_pred, y_true,
                                 one_hot=False,
                                 classes=args.out_channels,
                                 ignore=args.ignore)
    iou = CustomLoss.multi_loss(y_pred, y_true,
                                one_hot=False,
                                classes=args.out_channels,
                                ignore=args.ignore,
                                fn='iou')
    with open(f'./results/{title}/{name}.txt', 'a') as f:
        # should not be 0,1 hard coded, but oh well
        compare_class(y_pred, y_true, unique_l=[1, 0], f=f)
        print(f'Dice is {dice}', file=f)
        print(f'Iou is {iou}', file=f)
    compare_class(y_pred, y_true, unique_l=[1, 0])
    print(f'Dice is {dice}')
    print(f'Iou is {iou}')
    if args.store_pred:
        store_results(y_pred=y_pred, y_true=y_true,
                      title=title, name=name)


def test_model(model, v_dl, ts_dl, args, name, device):
    if args.k_cross:
        for k in model:
            test_model_fn(model[k], ts_dl[k], f'{args.title}/{k}', name,
                          args, device)
    else:
        test_model_fn(model, ts_dl, args.title, name,  args, device)
        v_dl.dataset.validate()
        test_model_fn(model, v_dl, args.title, name+'_val',  args, device,
                      val=True)


def test_per_dataset(model, test_dls, val_dls, args, name, device):
    for dname in test_dls:
        test_model(model, val_dls[dname], test_dls[dname],
                   args, name+'_'+dname, device)


def init_folder(title, add_time=True):
    mkdir('./results')
    if add_time:
        title = title + '_' + str(int(time.time()))
    if title in os.listdir('./results'):
        gen = np.random.default_rng()
        title += f'_{gen.integers(42000)}'
    mkdir('./results/'+title)
    return title


def save_args(args):
    store_results(**vars(args), name='args')


###
###
###
#  MAIN
###
###
###
def main(args):
    args.title = init_folder(args.title, add_time=False)
    log(f'Title: {args.title}', args.log, 1)
    save_args(args)
    device = torch.device("cpu" if not torch.cuda.is_available()
                          else args.device)
    log(f'Device: {device}', args.log, 1)
    log('Create dataset, dataloader, model', args.log, 1)
    tr_dl, v_dl, ts_dl, ds = get_data_loader(args)
    if args.k_cross:
        for i in tr_dl:
            init_folder(f'{args.title}/{i}', add_time=False)
    model = get_model(args, ds.in_channels, ds.img_size, device)
    model.to(device)
    optimizer = get_optimizer(args, model)
    loss_fn = CustomLoss(classes=args.out_channels,
                         ignore=args.ignore)
    model, l_tt, l_vt = dl_helper.train(
        args.k_cross,
        tr_dl, v_dl, model, loss_fn,
        optimizer, grad_acc=args.grad_acc, crop=args.tf_crop_test,
        title=args.title, log=args.log, save=args.save,
        epochs=args.md_epochs, device=device, early_stop=args.early_stop)
    if args.per_dataset:
        test_dls = make_dl_each_dataset(ds, key='test')
        val_dls = make_dl_each_dataset(ds, key='val')
        test_per_dataset(model, test_dls, val_dls, args, 'Last_model', device)
    test_model(model, v_dl, ts_dl, args,  'Last_model', device)
    if args.md_epochs > 0:
        best_model = get_best_model(tr_dl, args, device)
        test_model(best_model, v_dl, ts_dl, args, 'Best_model',
                   device)
    store_results(l_tt=l_tt, l_vt=l_vt, title=args.title, name='loss')


def get_optimizer(args, model):
    if args.md_optim == 'adam':
        optim_fn = torch.optim.Adam
    elif args.md_optim == 'adamw':
        optim_fn = torch.optim.AdamW
    if args.k_cross:
        optimizer = partial(optim_fn, lr=args.md_learning_rate,
                            weight_decay=args.md_weight_decay)
    else:
        optimizer = optim_fn(model.parameters(),
                             lr=args.md_learning_rate,
                             weight_decay=args.md_weight_decay)
    return optimizer


def get_best_model(tr_dl, args, device):
    if args.k_cross:
        res = {}
        for k in tr_dl.keys():
            model = get_model(args, tr_dl[k].dataset.in_channels,
                              tr_dl[k].dataset.img_size,  device)
            res[k] = dl_helper.load_model(f'{args.title}/{k}', model,
                                          device=device)
        return res
    else:
        model = get_model(args, tr_dl.dataset.in_channels,
                          tr_dl.dataset.img_size,  device)
        return dl_helper.load_model(args.title, model, device=device)


def get_model(args, in_channels, img_size, device):
    md = MD_DICT[args.md_model](init_features=args.md_features,
                                in_channels=in_channels,
                                out_channels=args.out_channels,
                                img_size=img_size,
                                n_block=args.md_block)
    if args.weights is not None:
        print('load pre trained')
        md = dl_helper.load_weights(args.weights, md, device=device)
    if args.k_cross:
        torch.save(md.state_dict(), f'./results/{args.title}/weights.pt')
    if len(args.md_freeze) > 0:
        md.freeze(args.md_freeze, nc=args.md_freeze_block)
    if len(args.md_unfreeze) > 0:
        md.freeze(args.md_unfreeze, nc=False, grad=True)
    # md.display_grad_per_block()
    return md


def make_dl_each_dataset(ds, key='test'):
    res = {}
    tmp = 0
    for i, name in enumerate(ds.dataset_name):
        length = len([j[1] for j in ds.type_indices[key] if j[0] == i])
        dl = DataLoader(
            ds,
            batch_size=args.dl_batch_size,
            sampler=list(range(tmp, tmp+length)))
        tmp += length
        res[name] = dl
    return res


def create_sampler(idx, args):
    res = list(range(idx))
    return res


def create_dl(dataset, args, idx, shu=True):
    # sampler = create_sampler(idx, args)
    dl = DataLoader(
        dataset,
        batch_size=args.dl_batch_size,
        shuffle=args.dl_shuffle and shu,
        # sampler=sampler,
        collate_fn=my_collate if args.grad_acc else None)
    return dl


def get_data_loader(args):
    dataset = get_dataset(args)
    if args.k_cross:
        tr_dl = {i: create_dl(dataset, args, len(v))
                 for i, v in enumerate(dataset.type_indices['train'])}
        # unique test
        ts_dl = {i: create_dl(dataset, args, len(v))
                 for i, v in enumerate(dataset.type_indices['test'])}
        v_dl = {i: create_dl(dataset, args, len(v))
                for i, v in enumerate(dataset.type_indices['val'])}
    else:
        tr_dl = create_dl(dataset, args, len(dataset.type_indices['train']))
        ts_dl = create_dl(dataset, args, len(dataset.type_indices['test']),
                          shu=False)
        v_dl = create_dl(dataset, args, len(dataset.type_indices['val']),
                         shu=False)
    return tr_dl, v_dl, ts_dl, dataset


def get_dataset(args):
    transforms = partial(
        get_transforms,
        angle=args.tf_angle,
        flip_prob=args.tf_flip,
        kernel=args.tf_kernel,
        sigma=args.tf_sigma,
        bright=args.tf_bright,  # tuple
        contrast=args.tf_contrast,  # tuple
        clip=args.tf_clip,
        invert_prob=args.tf_invert,
        erase_prob=args.tf_erase,
        erase_scale=args.tf_erase_scale,
        erase_ratio=args.tf_erase_ratio,
        erase_value=args.tf_erase_value,
        crop_prob=args.tf_crop,
        crop_size=args.tf_crop_size,
        min_0=args.tf_bright_min_0
    )
    test_transforms = get_test_transforms(
        pad_size=args.tf_pad_size,
        padding=args.tf_padding,
        pad_value=args.tf_pad_value,
        crop_size=args.tf_crop_size[0] if args.tf_crop_test else None
    )
    dataset = DatasetHandler(
        transforms=transforms,
        test_transforms=test_transforms,
        train_folder=args.dl_train_folder,
        datasets=args.dataset,
        path=args.path + ('' if args.size == '' else '_'+args.size),
        pkl_file=args.no_pkl_file,
        x_label=args.x_label,
        y_label=args.y_label,
        test_split=args.dl_test_subset,
        val_split=args.dl_val_subset,
        train_size=args.dl_train_size_subset,
        cross_nb=args.cross_nb if args.k_cross else None,
        split_shuffle=args.dl_split_shuffle,
        cross_key=args.dl_cross_key,
        seed=args.split_seed,
    )
    return dataset


if __name__ == '__main__':
    args = parse_args('OCTA dl')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
