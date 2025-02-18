import numpy as np
import torch
import torch.nn as nn


class MultiTaskLossVkitti(nn.Module):
    depth_class_range = [6, 8, 10, 14, 18, 23, 31, 40, 49, 59, 92, 147, 236, 655]

    def __init__(self, device, reverse=False):
        """

        :rtype: nn.Module
        """
        super(MultiTaskLossVkitti, self).__init__()
        self.device = device

        self.reverse = reverse

        self.loss_main_begin = None

        return


    def forward(self, masks_pred, deps_pred, true_masks, true_deps):
        # ======= Only this part is varying by task =======
        if self.reverse:
            criterion_seg = nn.CrossEntropyLoss(reduction='none')
            loss_seg_mat = criterion_seg(masks_pred, true_masks)  # (B, H, W)
            loss_main_mat = loss_seg_mat
        else:
            criterion_dep = nn.L1Loss(reduction='none')
            loss_dep_mat = criterion_dep(deps_pred, true_deps)  # (B, 1, H, W)
            loss_dep_mat = torch.squeeze(loss_dep_mat, dim=1)  # (B, H, W)
            loss_main_mat = loss_dep_mat
        # =================================================

        loss_main = torch.mean(loss_main_mat)

        if type(self.loss_main_begin)==type(None):
            self.loss_main_begin = loss_main.clone().detach()
            self.loss_main_begin = self.loss_main_begin.to(device=self.device, dtype=torch.float32)
        loss = loss_main/self.loss_main_begin #0.810534*loss_seg/self.loss_seg_begin + loss_dep/self.loss_dep_begin
        return loss


