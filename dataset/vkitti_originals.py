from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir='data/imgs/', masks_dir='data/masks/', deps_dir='data/depths/', scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.deps_dir = deps_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.worlds = ['0001/', '0002/', '0006/', '0018/', '0020/'] # input images parent directories
        self.variations = ['15-deg-left/', '15-deg-right/', '30-deg-left/', '30-deg-right/',
                            'clone/', 'fog/', 'morning/', 'overcast/', 'rain/', 'sunset/'] # input images directories
        self.categories = {0: [210, 0, 200], 1: [90, 200, 255], 2: [0, 199, 0], 3: [90, 240, 0], 4: [140, 140, 140],
                           5: [100, 60, 100], 6: [255, 100, 255], 7: [255, 255, 0], 8: [200, 200, 0], 9: [255, 130, 0],
                           10: [80, 80, 80], 11: [160, 60, 60]}
        # category: index(key)-RGB(value) mapping dictionary
        #'Terrain': 0, 'Sky': 1, 'Tree': 2, 'Vegetation': 3, 'Building': 4,
        #'Road': 5, 'GuardRail': 6, 'TrafficSign': 7, 'TrafficLight': 8, 'Pole': 9,
        #'Misc': 10, 'Truck': 11, 'Car': 12(Rest)
        self.ids = [] # list for input images' addresses
        for world in self.worlds:
            for variation in self.variations:
                self.ids += [world + variation + splitext(file)[0] for file in listdir(masks_dir + world + variation)
                              if not file.startswith('.')]

        logging.info('Creating dataset with {} examples'.format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    def preprocess(self, pil_img, scale, is_mask=False, is_depth=False):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h) # size of resized pil_img
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img) # pillow image -> numpy

        if len(img_nd.shape) == 2: # for depth image
            img_nd = np.expand_dims(img_nd, axis=2) # (H, W)-> (H, W, 1)

        if is_mask: # for masks
            img_temp = np.zeros((img_nd.shape[0],img_nd.shape[1]), dtype = np.int8) + 12 #newly generated (H, W) all pixel indexed as 12 : car
            for h in range(img_nd.shape[0]):
                for w in range(img_nd.shape[1]):
                    for cat_idx, rgb in self.categories.items():
                        if np.array_equal(rgb, img_nd[h][w]):
                            img_temp[h][w] = cat_idx # set index in img_temp
            img_trans = img_temp # (H, W)
        else: # for depth and input image
            img_trans = img_nd.transpose((2, 0, 1)) # (H, W, C) -> (C, H, W)
            if img_trans.max() > 1:
                if is_depth==True: # for depths
                    img_trans = np.reciprocal(img_trans, dtype=np.float32 )*100 # inverse depth image: element range from 1/655.35 ~ 1/3.xx or 1/4.xx
                else: # for rgb images
                    img_trans = img_trans / 255 # Normalize input image

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')
        dep_file = glob(self.deps_dir + idx + '*')

        assert len(mask_file) == 1, \
            'Either no image or multiple images found for the ID {}: {}'.format(idx, img_file)
        assert len(mask_file) == 1, \
            'Either no mask or multiple masks found for the ID {}: {}'.format(idx, mask_file)
        assert len(dep_file) == 1, \
            'Either no depth or multiple depths found for the ID {}: {}'.format(idx, dep_file)
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
        dep = Image.open(dep_file[0])

        assert img.size == mask.size, \
            'Image and mask {} not the same size: {} and {}'.format(idx, img.size, mask.size)
        assert img.size == dep.size, \
            'Image and depth {} not the same size: {} and {}'.format(idx, img.size, dep.size)
        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        dep = self.preprocess(dep, self.scale, is_depth=True)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask), 'depth': torch.from_numpy(dep)}
