import os
import torch
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, KFold
from torchvision import transforms
from PIL import Image

FUNDUS_PATH = '../data/retinal_vessel/fundus/img_less_fundus'
OCTA_PATH = '../data/retinal_vessel/octa/octa_output'


class DatasetHandler(torch.utils.data.Dataset):
    def __init__(
            self,
            datasets=None,
            path=OCTA_PATH,
            transforms=None,
            test_transforms=None,
            x_label='',
            y_label='',
            test_split=0.3,
            val_split=0.3,
            train_size=None,
            train_folder=True,
            cross_nb=None,
            cross_key='val',
            split_shuffle=True,
            seed=42,
            pkl_file=True,
            ):
        if datasets is None:
            datasets = 'all'
        self.seed = seed
        self.transforms = transforms
        self.test_transforms = test_transforms
        self.x_label = x_label if x_label == '' else '_' + x_label
        self.y_label = y_label if y_label == '' else '_' + y_label
        if test_split is not None and test_split > 1:
            self.test_split = int(test_split)
        else:
            self.test_split = test_split
        if val_split is not None and val_split > 1:
            self.val_split = int(val_split)
        else:
            self.val_split = val_split
        if train_size is not None and train_size > 1:
            self.train_size = int(train_size)
        else:
            self.train_size = train_size
        self.train_folder = train_folder
        self.cross_nb = cross_nb
        self.cross_key = cross_key
        self.split_shuffle = split_shuffle

        self.dataset_name = []
        self.transforms_list = []
        self.test_transforms_list = []
        self.x = []
        self.y = []
        self.means = []
        self.stds = []
        self.indices = []
        self.type_indices = {'train': [], 'val': [], 'test': []}
        if self.cross_nb is not None:
            for k in self.type_indices:
                self.type_indices[k] = [[] for i in range(self.cross_nb)]
        self.datasets_indices = []

        self.state = 'train'
        if self.cross_nb is not None:
            self.fold = 0

        self.load_datasets(path, datasets, pkl_file)
        self.update_indices()

        self.in_channels = self.x[0].shape[1]
        self.img_size = self.x[0].shape[2]

    def update_indices(self):
        self.current_indice = self._get_indices_state(self.state)

    def _get_indices_state(self, state):
        if self.cross_nb is not None:
            return self.type_indices[state][self.fold]
        else:
            return self.type_indices[state]

    def _get_datasets_indices_state(self, idx, state):
        if self.cross_nb is not None:
            return self.datasets_indices[idx][state][self.fold]
        else:
            return self.datasets_indices[idx][state]

    def validate(self):
        self.state = 'val'
        self.update_indices()

    def train(self):
        self.state = 'train'
        self.update_indices()

    def test(self):
        self.state = 'test'
        self.update_indices()

    def next_fold(self):
        if self.cross_nb is not None and self.fold < (self.cross_nb-1):
            self.fold += 1
            self.update_indices()

    def read_pkl_data(self, path):
        with open(f'{path}/x{self.x_label}.pkl', 'rb') as f:
            x = pickle.load(f)
        with open(f'{path}/y{self.y_label}.pkl', 'rb') as f:
            y = pickle.load(f)
        return x, y

    # put as static ?
    # or need info from self to process properly ?
    def process_torch_from_pil(self, img):
        return img / 255

    def read_img_folder(self, path):
        res = []
        for i in os.listdir(path):
            res.append(self.process_torch_from_pil(
                transforms.functional.pil_to_tensor(
                    Image.open(f'{path}/{i}'))))
        return torch.stack(res)

    def read_img_data(self, path):
        x = self.read_img_folder(f'{path}/x{self.x_label}')
        y = self.read_img_folder(f'{path}/y{self.y_label}')
        return x, y

    def read_data(self, path, pkl_file):
        if pkl_file:
            return self.read_pkl_data(path)
        else:
            return self.read_img_data(path)

    def load_dataset(self, path, pkl_file):
        lst = os.listdir(path)
        x = {}
        y = {}
        idx = {}
        # read data
        if 'train' in lst:
            x['train'], y['train'] = self.read_data(f'{path}/train', pkl_file)
            x['test'], y['test'] = self.read_data(f'{path}/test', pkl_file)
            if 'val' in lst:
                x['val'], y['val'] = self.read_data(f'{path}/val', pkl_file)
            if self.train_folder:
                j = 0
                for k in x:
                    idx[k] = list(range(j, j+len(x[k])))
                    j += len(x[k])
            for k in list(x.keys()):
                if k != 'train':
                    x['train'] = torch.cat((x['train'], x[k]))
                    y['train'] = torch.cat((y['train'], y[k]))
                    del x[k]
                    del y[k]
            x = x['train']
            y = y['train']
            if not self.train_folder:
                idx['train'] = list(range(len(x)))
        else:
            x, y = self.read_data(path, pkl_file)
            idx['train'] = list(range(len(x)))
        return x, y, idx

    def split_dataset_fixed(self, idx):
        if 'test' not in idx:
            if self.test_split == 1:
                idx['test'] = idx['train']
                idx['train'] = []
            else:
                idx['train'], idx['test'] = train_test_split(
                    idx['train'], test_size=self.test_split,
                    shuffle=self.split_shuffle, random_state=self.seed)
        if 'val' not in idx:
            if self.val_split == 0:
                idx['val'] = []
            else:
                idx['train'], idx['val'] = train_test_split(
                    idx['train'], test_size=self.val_split,
                    shuffle=self.split_shuffle, random_state=self.seed)
        if self.train_size is not None:
            choice = np.random.choice(idx['train'], self.train_size)
            idx['train'] = list(choice)
        return idx

    def split_dataset_cross(self, idx):
        kf = KFold(n_splits=self.cross_nb, shuffle=self.split_shuffle,
                   random_state=self.seed)
        if self.cross_key == 'val':
            if 'val' in idx:
                idx['train'].extend(idx['val'])
                del idx['val']
            if 'test' not in idx:
                idx['train'], idx['test'] = train_test_split(
                    idx['train'], test_size=self.test_split,
                    shuffle=self.split_shuffle)
            idx['test'] = [idx['test']] * self.cross_nb
            tmp = idx['train']
            idx['train'] = []
            idx['val'] = []
            for train, val in kf.split(tmp):
                idx['train'].append(train)
                idx['val'].append(val)
        else:
            if 'val' in idx:
                idx['train'].extend(idx['val'])
                del idx['val']
            if 'test' in idx:
                idx['train'].extend(idx['test'])
                del idx['test']
            tmp = idx['train']
            idx['train'] = []
            idx['test'] = []
            idx['val'] = []
            for train, test in kf.split(tmp):
                idx['test'].append(test)
                if self.val_split == 0:
                    val = []
                else:
                    train, val = train_test_split(
                        train, test_size=self.val_split,
                        shuffle=self.split_shuffle)
                idx['train'].append(train)
                idx['val'].append(val)
        return idx

    def split_dataset(self, idx):
        if self.cross_nb is not None:
            return self.split_dataset_cross(idx)
        else:
            return self.split_dataset_fixed(idx)

    def add_indices(self, idx, idx_d):
        if self.cross_nb is None:
            for k in idx:
                self.type_indices[k].extend([(idx_d, i) for i in idx[k]])
        else:
            for k in idx:  # key state
                for n, fold in enumerate(idx[k]):
                    # type_indices is {'train: [idx1, idx2, ...], ...}
                    self.type_indices[k][n].extend([(idx_d, i)
                                                    for i in fold])
        self.datasets_indices.append(idx)

    def process_dataset(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.test_split == 1:
            train_idx = self._get_datasets_indices_state(idx, 'test')
        else:
            train_idx = self._get_datasets_indices_state(idx, 'train')
        mean = x[train_idx].mean()
        std = x[train_idx].std()
        print('- PR - Normalized with a fixed'
              + f' mean ({mean}) and std ({std})')
        x, y = DatasetHandler.process_data(
            x, y, mean, std)
        self.x[idx] = x
        self.y[idx] = y

    def add_transforms(self, idx):
        xmin = self.x[idx].min()
        xmax = self.x[idx].max()
        if self.transforms is not None:
            self.transforms_list.append(
                self.transforms(clip_l=xmin, clip_h=xmax))
        else:
            self.transforms_list.append(None)
        self.test_transforms_list.append(self.test_transforms)

    def add_dataset(self, path, name, pkl_file):
        print(f'- IN - Read {name} set')
        x, y, idx = self.load_dataset(path, pkl_file)
        self.x.append(x)
        self.y.append(y)
        # print(x.shape, y.shape)
        self.means.append(x.mean(dim=(0, 2, 3)))
        self.stds.append(x.std(dim=(0, 2, 3)))
        # print(x.mean(dim=(0, 2, 3)))
        idx = self.split_dataset(idx)
        j = len(self.dataset_name)
        self.add_indices(idx, j)
        self.indices.extend([(j, i) for i in range(len(x))])
        self.dataset_name.append(name)
        self.process_dataset(j)
        self.add_transforms(j)

    def load_datasets(self, path, ds, pkl_file):
        if isinstance(ds, str):
            if ds == 'all':
                ds_name = os.listdir(path)
                for i in ds_name:
                    self.add_dataset(f'{path}/{i}', i, pkl_file)
            else:
                self.add_dataset(f'{path}/{ds}', ds, pkl_file)
        elif isinstance(ds, list):
            for i in ds:
                self.add_dataset(f'{path}/{i}', i, pkl_file)

    def __len__(self):
        return len(self.type_indices[self.state])

    def __getitem__(self, idx):
        dsn, i = self.current_indice[idx]

        x = self.x[dsn][i]
        y = self.y[dsn][i]

        if self.state == 'train' and self.transforms_list[dsn] is not None:
            x, y = self.transforms_list[dsn]((x, y))

        if (self.state != 'train'
           and self.test_transforms_list[dsn] is not None):
            x, y = self.test_transforms_list[dsn]((x, y))

        return x, y

    def process_data(x, y, mean, std):
        x = transforms.functional.normalize(
                x, mean=mean, std=std)
        x = x.float()
        y = (y > 0.15).int()
        return x, y
