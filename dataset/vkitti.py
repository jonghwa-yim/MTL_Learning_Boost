__author__ = "Jonghwa Yim"
__credits__ = ["Jonghwa Yim"]
__copyright__ = "Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved"
__email__ = "jonghwa.yim@samsung.com"

import os

import numpy as np
import torch
import torch.utils.data as data


class DatasetFromTXT(data.Dataset):
    def __init__(self, txt_file, imgs_dir='data/imgs/', masks_dir='data/masks/', deps_dir='data/depths/'):
        super(DatasetFromTXT, self).__init__()
        self.dataset_root = '/f_data2/TrainingSets/vkitti/data_numpy/'
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.deps_dir = deps_dir

        fp = open(txt_file, 'r')
        self.image_filenames = fp.readlines()
        fp.close()
        pass

    def __getitem__(self, index):
        filenames = self.image_filenames[index]
        img_file = os.path.join(self.dataset_root, self.imgs_dir, filenames.rstrip('\n') + '.npy')
        mask_file = os.path.join(self.dataset_root, self.masks_dir, filenames.rstrip('\n') + '.npy')
        dep_file = os.path.join(self.dataset_root, self.deps_dir, filenames.rstrip('\n') + '.npy')

        img_np = np.load(img_file).astype(np.float32)
        mask_np = np.load(mask_file).astype(np.int)
        dep_np = np.load(dep_file).astype(np.float32)

        return {'image': torch.from_numpy(img_np), 'mask': torch.from_numpy(mask_np), 'depth': torch.from_numpy(dep_np)}

    def __len__(self):
        return len(self.image_filenames)