__author__ = "Jonghwa Yim"
__credits__ = ["Jonghwa Yim"]
__copyright__ = "Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved"
__email__ = "jonghwa.yim@samsung.com"

import os

import numpy as np
import torch.utils.data as data
import torch
import skimage.transform


class NYUV2Dataset(data.Dataset):
    def __init__(self, trainval='train', img_size=256):
        """
        NYU v2 Dataset.
        Class label ignore value is -1. and range from 0 to 12
        Depth ignore value is 0. and value means distance. range from 0 to 9.99
        :param trainval:
        :param img_size:
        """
        super(NYUV2Dataset, self).__init__()
        self.dataset_root = '/f_data1/TrainingSets/nyuv2/'

        self.imgs_dir = 'image'
        self.masks_dir = 'label'
        self.deps_dir = 'depth'
        self.trainval = trainval + '/'
        self.img_size = img_size

        self.ids = [] # list for input images' addresses

        for file in os.listdir(self.dataset_root + self.trainval + self.imgs_dir):
            if not file.startswith('.'):
                self.ids += [file.split('.')[0]]
        pass

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = self.dataset_root + self.trainval + self.imgs_dir + '/' + idx + '.npy'
        mask_file = self.dataset_root + self.trainval + self.masks_dir + '/' + idx + '.npy'
        dep_file = self.dataset_root + self.trainval + self.deps_dir + '/' + idx + '.npy'

        img_np = np.load(img_file).astype(np.float32)
        mask_np = np.load(mask_file).astype(np.int) + 1 # to make ignore index 0, not -1. So that the ignore index same as that of depth
        dep_np = np.load(dep_file).astype(np.float32)

        img_np = skimage.transform.resize(img_np, (self.img_size, self.img_size), mode='reflect', anti_aliasing=False).astype(np.float32)
        mask_np = skimage.transform.resize(mask_np, (self.img_size, self.img_size), order=0, mode='reflect', anti_aliasing=False, preserve_range=True).astype(np.int)
        dep_np = skimage.transform.resize(dep_np, (self.img_size, self.img_size), mode='reflect', anti_aliasing=False).astype(np.float32)

        img_np = img_np.transpose(2,0,1)
        dep_np = dep_np.transpose(2,0,1)

        return {'image': torch.from_numpy(img_np), 'mask': torch.from_numpy(mask_np), 'depth': torch.from_numpy(dep_np)}


if __name__ == '__main__':
    dataset = NYUV2Dataset()

    minval = 0
    maxval = 0
    minval_d = 0
    maxval_d = 0
    for item in dataset:
        if item['mask'].max() > maxval:
            maxval = item['mask'].max()
        if item['mask'].min() < minval:
            minval = item['mask'].min()

        if item['depth'].max() > maxval_d:
            maxval_d = item['depth'].max()
        if item['depth'].min() < minval_d:
            minval_d = item['depth'].min()

    print('minval_seg : ', minval)
    print('maxval_seg : ', maxval)
    print('minval_dep : ', minval_d)
    print('maxval_dep : ', maxval_d)
    print('done')
