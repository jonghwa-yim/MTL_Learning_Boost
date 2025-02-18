import numpy as np
import torch
import torch.nn as nn


class MultiTaskLossNYU(nn.Module):
    depth_class_range = [1,2,3,4,5,6,7,8,9,10]

    def __init__(self, device, reverse=False):
        """

        :rtype: nn.Module
        """
        super(MultiTaskLossNYU, self).__init__()
        self.device = device

        self.reverse = reverse

        self.loss_main_begin = None

        return


    def forward(self, masks_pred, deps_pred, true_masks, true_deps):

        # ======= Only this part is varying by task =======
        if self.reverse:
            criterion_seg = nn.CrossEntropyLoss(reduction='none', ignore_index=0)

            seg_active_mask = (true_masks != 0)
            seg_active_pixel_num = torch.sum(seg_active_mask)

            loss_seg_mat = criterion_seg(masks_pred, true_masks)  # (B, H, W)

            loss_main_mat = loss_seg_mat
            main_pixel_num = seg_active_pixel_num
        else:
            criterion_dep = nn.L1Loss(reduction='none')

            deps_active_mask = (true_deps != 0)
            deps_active_pixel_num = torch.sum(deps_active_mask)  # .detach().cpu()
            deps_pred = deps_active_mask.type(torch.float) * deps_pred

            loss_dep_mat = criterion_dep(deps_pred, true_deps)  # (B, 1, H, W)
            loss_dep_mat = torch.squeeze(loss_dep_mat, dim=1)  # (B, H, W)

            loss_main_mat = loss_dep_mat
            main_pixel_num = deps_active_pixel_num
        # =================================================

        loss_main = torch.sum(loss_main_mat) / main_pixel_num

        if type(self.loss_main_begin)==type(None):
            self.loss_main_begin = loss_main.clone().detach()
            self.loss_main_begin = self.loss_main_begin.to(device=self.device, dtype=torch.float32)
        loss = loss_main/self.loss_main_begin #0.810534*loss_seg/self.loss_seg_begin + loss_dep/self.loss_dep_begin
        return loss

