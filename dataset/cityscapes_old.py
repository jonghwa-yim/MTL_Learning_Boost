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
    def __init__(self, phase='train', img_size=256):
        """
        NYU v2 Dataset.
        Class label ignore value is -1. and range from 0 to 12
        Depth ignore value is 0. and value means distance. range from 0 to 9.99
        :param trainval:
        :param img_size:
        """
        super(CITISCAPESDataset, self).__init__()
        self.dataset_root = '/f_data1/TrainingSets/Cityscapes_Dataset/'

        self.imgs_dir = 'leftImg8bit_trainvaltest/leftImg8bit/'
        self.masks_dir = 'gtFine_trainvaltest/gtFine/'
        self.deps_dir = 'disparity_trainvaltest/disparity/'
        self.trainval = phase + '/'
        self.img_size = img_size

        self.ids = [] # list for input images' addresses

        if phase == 'train':
            path1 = self.dataset_root + self.imgs_dir + 'train/'
            for folder2 in os.listdir(path1):
                for file in os.listdir(os.path.join(path1, folder2)):
                    if not file.startswith('.'):
                        self.ids += [folder2 + '/' + file.split('leftImg8bit')[0]]

        elif phase == 'val':
            path1 = self.dataset_root + self.imgs_dir + 'val/'
            for folder2 in os.listdir(path1):
                for file in os.listdir(os.path.join(path1, folder2)):
                    if not file.startswith('.'):
                        self.ids += [folder2 + '/' + file.split('leftImg8bit')[0]]

        elif phase == 'test':
            path1 = self.dataset_root + self.imgs_dir + 'test/'
            for folder2 in os.listdir(path1):
                for file in os.listdir(os.path.join(path1, folder2)):
                    if not file.startswith('.'):
                        self.ids += [folder2 + '/' + file.split('leftImg8bit')[0]]

        pass

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]

        img_file = glob(self.dataset_root + self.imgs_dir + self.trainval + idx + '*')

        input_img = default_loader(img_file[0])
        input_img = input_img.resize((256, 256))

        disp_file = glob(self.dataset_root + self.deps_dir + self.trainval + idx + '*')
        input_disp = Image.open(disp_file[0])
        input_disp = input_disp.resize((256, 256))
        disp_t = np.asarray(input_disp, np.float)
        disp_t2 = np.subtract(disp_t, 1., where=disp_t!=0.) / 256.
        dispm = disp_t2 >= 1.
        disp_t2 = disp_t2 * dispm

        seg_file = glob(self.dataset_root + self.masks_dir + self.trainval + idx + 'gtFine_label*')
        seg_img = Image.open(seg_file[0])
        seg_img = seg_img.resize((256, 256), resample=0)    # nearest resize
        seg_img = np.asarray(seg_img)

        return input_img, disp_t2, seg_img, idx

        target_img = default_loader(idx)


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

    max_dep = 0
    min_seg = 1
    max_seg = 0

    dataset = CITISCAPESDataset(phase='train')

    out_img_folder = '/f_data1/TrainingSets/Cityscapes_Dataset/train/img'
    out_dep_folder = '/f_data1/TrainingSets/Cityscapes_Dataset/train/depth'
    out_seg_folder = '/f_data1/TrainingSets/Cityscapes_Dataset/train/seg'

    for item in dataset:
        input_img, disp_t, seg_img, fname = item

        if disp_t.max() > max_dep:
            max_dep = disp_t.max()

        if seg_img.max() > max_seg:
            max_seg = seg_img.max()
        if seg_img.min() < min_seg:
            min_seg = seg_img.min()

        #input_img.save(out_img_folder + '/' + fname.split('/')[1] + 'train.png')
        #np.save(out_seg_folder + '/' + fname.split('/')[1] + 'train.npy', seg_img)
        #np.save(out_dep_folder + '/' + fname.split('/')[1] + 'train.npy', disp_t)

    print('minval_seg : ', min_seg)
    print('maxval_seg : ', max_seg)
    print('maxval_dep : ', max_dep)
    # ===========================================================================
    max_dep = 0
    min_seg = 1
    max_seg = 0
    dataset = CITISCAPESDataset(phase='val')

    out_img_folder = '/f_data1/TrainingSets/Cityscapes_Dataset/train/img'
    out_dep_folder = '/f_data1/TrainingSets/Cityscapes_Dataset/train/depth'
    out_seg_folder = '/f_data1/TrainingSets/Cityscapes_Dataset/train/seg'

    for item in dataset:
        input_img, disp_t, seg_img, fname = item

        if disp_t.max() > max_dep:
            max_dep = disp_t.max()

        if seg_img.max() > max_seg:
            max_seg = seg_img.max()
        if seg_img.min() < min_seg:
            min_seg = seg_img.min()

        #input_img.save(out_img_folder + '/' + fname.split('/')[1] + 'val.png')
        #np.save(out_seg_folder + '/' + fname.split('/')[1] + 'val.npy', seg_img)
        #np.save(out_dep_folder + '/' + fname.split('/')[1] + 'val.npy', disp_t)

    print('minval_seg : ', min_seg)
    print('maxval_seg : ', max_seg)
    print('maxval_dep : ', max_dep)

    # ===========================================================================
    max_dep = 0
    min_seg = 1
    max_seg = 0
    dataset = CITISCAPESDataset(phase='test')

    out_img_folder = '/f_data1/TrainingSets/Cityscapes_Dataset/test/img'
    out_dep_folder = '/f_data1/TrainingSets/Cityscapes_Dataset/test/depth'
    out_seg_folder = '/f_data1/TrainingSets/Cityscapes_Dataset/test/seg'

    for item in dataset:
        input_img, disp_t, seg_img, fname = item

        if disp_t.max() > max_dep:
            max_dep = disp_t.max()

        if seg_img.max() > max_seg:
            max_seg = seg_img.max()
        if seg_img.min() < min_seg:
            min_seg = seg_img.min()

        #input_img.save(out_img_folder + '/' + fname.split('/')[1] + 'test.png')
        #np.save(out_seg_folder + '/' + fname.split('/')[1] + 'test.npy', seg_img)
        #np.save(out_dep_folder + '/' + fname.split('/')[1] + 'test.npy', disp_t)

    print('minval_seg : ', min_seg)
    print('maxval_seg : ', max_seg)
    print('maxval_dep : ', max_dep)
    print('done')
