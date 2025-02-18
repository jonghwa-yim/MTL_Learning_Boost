__author__ = "Jonghwa Yim"
__credits__ = ["Jonghwa Yim"]
__copyright__ = "Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved"
__email__ = "jonghwa.yim@samsung.com"

import os

import numpy as np
import torch.utils.data as data
import torch
import skimage.transform
from PIL import Image
from glob import glob
import torchvision.transforms as transforms


def input_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CITISCAPESDataset(data.Dataset):
    def __init__(self, phase='train'):
        """
        Citiscapes Dataset.
        Class label ignore value is 0. and range from 1 to 33
        Depth disparity. ignore value is 0. and value means distance. range from 1 to 126.004
        :param phase: 'train' or 'val'
        """
        super(CITISCAPESDataset, self).__init__()
        assert phase == 'train' or 'val'

        self.dataset_root = '/f_data1/TrainingSets/Cityscapes_Dataset/'

        self.imgs_dir = 'img'
        self.masks_dir = 'seg'
        self.deps_dir = 'depth'
        self.phase = phase + '/'
        self.img_size = 256

        self.ids = [] # list for input images' addresses

        for file in os.listdir(self.dataset_root + self.phase + self.imgs_dir):
            if not file.startswith('.'):
                self.ids += [file.split('.')[0]]

        pass

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = self.dataset_root + self.phase + self.imgs_dir + '/' + idx + '.png'
        mask_file = self.dataset_root + self.phase + self.masks_dir + '/' + idx + '.npy'
        dep_file = self.dataset_root + self.phase + self.deps_dir + '/' + idx + '.npy'

        img_np = default_loader(img_file)
        mask_np = np.load(mask_file).astype(np.int)
        dep_np = np.load(dep_file).astype(np.float32)

        # img_np = skimage.transform.resize(img_np, (self.img_size, self.img_size), mode='reflect', anti_aliasing=False).astype(np.float32)
        # mask_np = skimage.transform.resize(mask_np, (self.img_size, self.img_size), order=0, mode='reflect', anti_aliasing=False, preserve_range=True).astype(np.int)
        # dep_np = skimage.transform.resize(dep_np, (self.img_size, self.img_size), mode='reflect', anti_aliasing=False).astype(np.float32)

        dep_np = dep_np.reshape((-1, dep_np.shape[0], dep_np.shape[1]))

        return {'image': input_transform()(img_np), 'mask': torch.from_numpy(mask_np), 'depth': torch.from_numpy(dep_np)}


if __name__ == '__main__':

    dataset = CITISCAPESDataset(phase='train')

    for item in dataset:
        input_img, disp_t, seg_img, fname = item

        if disp_t.max() > max_dep:
            max_dep = disp_t.max()

        if seg_img.max() > max_seg:
            max_seg = seg_img.max()
        if seg_img.min() < min_seg:
            min_seg = seg_img.min()

    print('minval_seg : ', min_seg)
    print('maxval_seg : ', max_seg)
    print('maxval_dep : ', max_dep)
    print('done')
