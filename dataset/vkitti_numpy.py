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
        # self.dataset_root = '/home/kshwan0227/MTL/Unet2/'
        self.dataset_root = '/f_data2/TrainingSets/vkitti/'
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
                self.ids += [world + variation + splitext(file)[0] for file in listdir(self.dataset_root + masks_dir + world + variation)
                              if not file.startswith('.')]

        logging.info('Creating dataset with {} examples'.format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.dataset_root + 'data_numpy/' + self.imgs_dir + idx + '*')
        mask_file = glob(self.dataset_root + 'data_numpy/' + self.masks_dir + idx + '*')
        dep_file = glob(self.dataset_root + 'data_numpy/' + self.deps_dir + idx + '*')

        assert len(img_file) == 1, \
            'Either no image or multiple images found for the ID {}: {}'.format(idx, img_file)
        assert len(mask_file) == 1, \
            'Either no mask or multiple masks found for the ID {}: {}'.format(idx, mask_file)
        assert len(dep_file) == 1, \
            'Either no depth or multiple depths found for the ID {}: {}'.format(idx, dep_file)

        img_np = np.load(img_file[0]).astype(np.float32)
        mask_np = np.load(mask_file[0]).astype(np.int)
        dep_np = np.load(dep_file[0]).astype(np.float32)

        return {'image': torch.from_numpy(img_np), 'mask': torch.from_numpy(mask_np), 'depth': torch.from_numpy(dep_np)}
