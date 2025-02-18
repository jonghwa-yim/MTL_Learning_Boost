""" Full assembly of the parts to form the complete network """
import torch
import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, reverse=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.reverse = reverse

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        if self.reverse:
            self.up_s1 = Up(1024, 256, bilinear)
            self.up_s2 = Up(512, 128, bilinear)
            self.up_s3 = Up(256, 64, bilinear)
            self.up_s4 = Up(128, 64, bilinear)
            self.out_seg = OutConv(64, n_classes)
        else:
            self.up_d1 = Up(1024, 256, bilinear)
            self.up_d2 = Up(512, 128, bilinear)
            self.up_d3 = Up(256, 64, bilinear)
            self.up_d4 = Up(128, 64, bilinear)
            self.out_dep = OutConv(64, 1) # 1 output channel for depth

    def forward(self, x):
        ret = dict()

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.reverse:
            #For semantic segmentation
            x = self.up_s1(x5, x4)
            x = self.up_s2(x, x3)
            x = self.up_s3(x, x2)
            x = self.up_s4(x, x1)
            segs = self.out_seg(x)
            ret = {
                'depth_prediction': None,
                'segmentation_prediction': segs,
            }
        else:
            #For depth detection
            y = self.up_d1(x5, x4)
            y = self.up_d2(y, x3)
            y = self.up_d3(y, x2)
            y = self.up_d4(y, x1)
            depths = self.out_dep(y)
            ret = {
                'depth_prediction': depths,
                'segmentation_prediction': None,
            }

        return ret  #torch.cat([depths, logits], dim=1) # size (C = n_classes + 1, H, W)
