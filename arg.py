import argparse


def parse_args_ds(argp):
    argp.add_argument('--n_img', type=int, default=-1,
                      help="Limit to the number of image to load",
                      dest='ds_n_img')
    return argp


def parse_args_dl(argp):
    argp.add_argument("--shuffle", action='store_true',
                      help="Flag to not shuffle when training (for epochs)",
                      dest='dl_shuffle')
    argp.set_defaults(dl_shuffle=False)
    argp.add_argument("--no_split_shuffle", action='store_false',
                      help="Flag to not shuffle when splitting",
                      dest='dl_split_shuffle')
    argp.set_defaults(dl_split_shuffle=True)
    argp.add_argument('--batch_size', type=int, default=1,
                      help="The bath size to compute the loss on",
                      dest='dl_batch_size')
    argp.add_argument("--no_train_folder", action='store_false',
                      help="Flag to not use the existing split folder",
                      dest='dl_train_folder')
    argp.set_defaults(no_train_folder=True)
    argp.add_argument('--test_subset', type=float,
                      help="Size for the training/testing split",
                      dest='dl_test_subset')
    argp.set_defaults(dl_test_subset=0.3)
    argp.add_argument('--val_subset', type=float,
                      help="Size for the training/validation split",
                      dest='dl_val_subset')
    argp.set_defaults(dl_val_subset=0.3)
    argp.add_argument('--train_size_subset', type=float,
                      help="Size for the training/validation after split",
                      dest='dl_train_size_subset')
    argp.set_defaults(dl_train_size_subset=None)
    argp.add_argument("--fixed_test_set", action='store_true',
                      help="Whether to fix the test set for CV",
                      dest='dl_fixed_test_set')
    argp.set_defaults(dl_fixed_test_set=True)
    argp.add_argument('--cross_key', type=str, choices=['val', 'test'],
                      default='val',
                      help="Dataset to use for the testing fold",
                      dest='dl_cross_key')
    return argp


def parse_args_md(argp):
    argp.add_argument('--epochs', type=int, default=1000,
                      help="The number of epochs to train the model",
                      dest='md_epochs')
    argp.add_argument('--learning_rate', type=float, default=1e-4,
                      help="The learning rate to which train the model",
                      dest='md_learning_rate')
    argp.add_argument('--weight_decay', type=float, default=0,
                      help="The weight decay to which train the model",
                      dest='md_weight_decay')
    argp.add_argument('--features', type=int, default=64,
                      help="The number of features for the conv layer",
                      dest='md_features')
    argp.add_argument('--model', type=str, default='unet_r',
                      help="The model to use for segmentation",
                      dest='md_model')
    argp.add_argument('--freeze', nargs='*', type=str,
                      help="which layer to freeze",
                      dest='md_freeze')
    argp.set_defaults(md_freeze=[])
    argp.add_argument('--unfreeze', nargs='*', type=str,
                      help="which layer to not freeze",
                      dest='md_unfreeze')
    argp.set_defaults(md_unfreeze=[])
    argp.add_argument('--freeze_layer', action='store_false',
                      help="freeze layer instead of block",
                      dest='md_freeze_block')
    argp.set_defaults(md_freeze_block=True)
    argp.add_argument('--n_block', type=int, default=4,
                      help="number of unet block",
                      dest='md_block')
    argp.add_argument('--optim', type=str, choices=['adam', 'adamw'],
                      help="the optimizer to use",
                      dest='md_optim')
    argp.set_defaults(md_optim='adam')
    return argp


def erase_value(arg):
    if arg == 'random':
        return arg
    else:
        return float(arg)


def parse_args_tf(argp):
    argp.add_argument('--angle', type=int,
                      help="Angle value for data augmentation",
                      dest='tf_angle')
    argp.add_argument('--no_angle', help='no angle', dest='tf_angle',
                      action='store_const', const=None)
    argp.set_defaults(tf_angle=180)
    argp.add_argument('--flip', type=float,
                      help="Flip probability value for data augmentation",
                      dest='tf_flip')
    argp.add_argument('--no_flip', help='no flip', dest='tf_flip',
                      action='store_const', const=None)
    argp.set_defaults(tf_flip=0.5)
    argp.add_argument('--invert', type=float,
                      help="Invert probability value for data augmentation",
                      dest='tf_invert')
    argp.add_argument('--no_invert', help='no invert', dest='tf_invert',
                      action='store_const', const=None)
    argp.set_defaults(tf_invert=0.5)
    argp.add_argument('--kernel', type=int,
                      help="Gaussian kernel value for data augmentation",
                      dest='tf_kernel')
    argp.add_argument('--no_kernel', help='no kernel', dest='tf_kernel',
                      action='store_const', const=None)
    argp.set_defaults(tf_kernel=3)
    argp.add_argument('--sigma', type=float, nargs=2,
                      help="Sigma gaussian blur values for data augmentation",
                      dest='tf_sigma')
    argp.add_argument('--no_sigma', help='no sigma', dest='tf_sigma',
                      action='store_const', const=None)
    argp.set_defaults(tf_sigma=[0., 1.])
    argp.add_argument('--bright', type=float, nargs=2,
                      help="Brightness high/low values or data augmentation",
                      dest='tf_bright')
    argp.add_argument('--no_bright', help='no bright', dest='tf_bright',
                      action='store_const', const=None)
    argp.set_defaults(tf_bright=[0.7, 1.3])
    argp.add_argument('--bright_min', action='store_false',
                      help="Flag to use min or 0 in bright. False is minimum",
                      dest='tf_bright_min_0')
    argp.set_defaults(tf_bright_min_0=True)
    argp.add_argument('--contrast', type=float, nargs=2,
                      help="Contrast high/low values for data augmentation",
                      dest='tf_contrast')
    argp.add_argument('--no_contrast', help='no contrast', dest='tf_contrast',
                      action='store_const', const=None)
    argp.set_defaults(tf_contrast=[0.7, 1.3])
    argp.add_argument('--clip', action='store_true',
                      help="Flag to clip values after bright/contrast blend",
                      dest='tf_clip')
    argp.add_argument('--crop', type=float,
                      help="Crop probability value for data augmentation",
                      dest='tf_crop')
    argp.add_argument('--yes_crop', help='yes crop', dest='tf_crop',
                      action='store_const', const=1)
    argp.set_defaults(tf_crop=0)
    argp.add_argument('--crop_size', nargs='+', type=int,
                      help="Crop size values for data augmentation."
                      + "1 or 2 value, for range or fixed",
                      dest='tf_crop_size')
    argp.set_defaults(tf_crop_size=[80])
    argp.add_argument('--erase', type=float,
                      help="Erase probability value for data augmentation",
                      dest='tf_erase')
    argp.add_argument('--no_erase', help='no erase', dest='tf_erase',
                      action='store_const', const=None)
    argp.set_defaults(tf_erase=0.5)
    argp.add_argument('--erase_scale', nargs=2, type=float,
                      help="Erase scale values for data augmentation",
                      dest='tf_erase_scale')
    argp.set_defaults(tf_erase_scale=[0.02, 0.15])
    argp.add_argument('--erase_ratio', nargs=2, type=float,
                      help="Erase ratio values for data augmentation",
                      dest='tf_erase_ratio')
    argp.set_defaults(tf_erase_ratio=[0.3, 3.3])
    argp.add_argument('--erase_value', type=erase_value,
                      help="Erase value values for data augmentation",
                      dest='tf_erase_value')
    argp.set_defaults(tf_erase_value='random')
    argp.add_argument('--crop_test', action='store_true',
                      help="Crop the testing images as well",
                      dest='tf_crop_test')
    argp.set_defaults(tf_crop_test=False)
    argp.add_argument('--pad_value', type=float,
                      help='The value to fill when padding',
                      dest='tf_erase')
    argp.set_defaults(tf_pad_value=0.)
    argp.add_argument('--pad_size', nargs='*', type=int,
                      help="Padding size values. Must me 3 values for BxHxW",
                      dest='tf_pad_size')
    argp.set_defaults(tf_pad_size=None)
    argp.add_argument('--padding', nargs='*', type=int,
                      help="Padding size values. Must me 3 couple"
                      + " (right and left) for BxHxW",
                      dest='tf_padding')
    argp.set_defaults(tf_padding=None)
    return argp


def choose_gpu(arg):
    if arg == 'cuda':
        return arg
    elif arg in ['0', '1', '2', '3']:
        return f'cuda:{arg}'
    else:
        raise argparse.ArgumentTypeError('GPU should be 0, 1, 2, or 3')


def check_tf(arg, parser):
    if arg.tf_pad_size is not None:
        if len(arg.tf_pad_size) != 3:
            raise parser.error(
                'Pad size should be 3 values for C, H, W')
    if arg.tf_padding is not None:
        if len(arg.tf_padding) != 6:
            raise parser.error(
                'Padding should be 6 values. Left and right for C, H, W')
    if arg.tf_crop_size is not None:
        if len(arg.tf_crop_size) not in [1, 2]:
            raise parser.error(
                'Crop size should be one value for fixed or 2 value for range')
    return arg


def choose_path(arg):
    if arg == 'fundus':
        return '../data/retinal_vessel/fundus/imgless_fundus'
    elif arg == 'fundusimg':
        return '../data/retinal_vessel/fundus/img_fundus_all'
    elif arg == 'octa':
        return '../data/retinal_vessel/octa/octa_output'
    elif arg == 'test':
        return '../data/retinal_vessel/fundus/orig_test_rgb'
    elif arg == 'testgreen':
        return '../data/retinal_vessel/fundus/orig_test_green'
    else:
        raise argparse.ArgumentTypeError('Path should be fundus or octa')


def parse_args(name):
    argp = argparse.ArgumentParser(name)
    argp.add_argument('--out', type=int, default=2,
                      help="How much class",
                      dest="out_channels")
    argp.add_argument("--k_cross", action='store_true',
                      help="Flag to add K-foldcross validation")
    argp.set_defaults(k_cross=False)
    argp.add_argument("--no_pkl_file", action='store_false',
                      help="Flag to not read data as pkl file")
    argp.set_defaults(no_pkl_file=True)
    argp.add_argument('--size', type=str, default='400',
                      help="The size of the image to load."
                      + "Can be 304, 320, 400 or orig")
    argp.add_argument('--path', type=choose_path,
                      default='fundus',
                      help="The path to the fundus folder")
    argp.add_argument('--x_label', type=str, default='',
                      help="Label to suffix to the x file name")
    argp.add_argument('--y_label', type=str, default='',
                      help="Label to suffix to the y file name")
    argp.add_argument('--dataset', type=str, nargs='*', default=None,
                      help="Name of the datasets to process")
    argp.add_argument("--cross_nb", type=int, default=5,
                      help="Integer representing how many fold for CV")
    argp.add_argument('--device', default='cuda', type=choose_gpu,
                      help="Which gpu to use. Default is all")
    argp.add_argument('--seed', type=int, default=42,
                      help="Seed")
    argp.add_argument('--split_seed', type=int, default=42,
                      help="Seed")
    argp.add_argument('--early_stop', type=int, default=400,
                      help="early stop")
    argp.add_argument('--log', choices=range(2), type=int,
                      default=1,
                      help="Log level. Can be 0 (nothing) or 1-2")
    argp.add_argument('--title', type=str, default="octa",
                      help="Title of the file to save the model in")
    argp.add_argument('--weights', type=str, default=None,
                      help="Name of the weight files to load")
    argp.add_argument('--ignore', type=int, default=0,
                      help="Ignore a label for dice computation")
    argp.add_argument("--grad_acc", action='store_true',
                      help="Flag to add gradient accumulation")
    argp.set_defaults(grad_acc=False)
    argp.add_argument("--store_pred", action='store_true',
                      help="Flag to store the predictions")
    argp.set_defaults(store_pred=False)
    argp.add_argument('--save', type=str, default='best',
                      choices=['last', 'best'],
                      help="Save whether the best model, or the last")
    argp.add_argument('--per_dataset', action='store_true',
                      help="Flag to compute metrics for each subdataset")
    argp.set_defaults(per_dataset=False)
    argp = parse_args_ds(argp)
    argp = parse_args_dl(argp)
    argp = parse_args_md(argp)
    argp = parse_args_tf(argp)
    arg = argp.parse_args()
    check_tf(arg, argp)
    return arg
