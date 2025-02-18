import numpy as np
import torch
import torch.nn as nn


class MultiTaskLossNYU(nn.Module):
    depth_class_range = [1,2,3,4,5,6,7,8,9,10]

    def __init__(self, n_classes, device, reverse=False):
        """

        :rtype: nn.Module
        """
        super(MultiTaskLossNYU, self).__init__()
        self.device = device

        # manual setting for classify the depth labels
        if reverse:
            self.n_classes = len(MultiTaskLossNYU.depth_class_range) + 1 # +1 including ignore index 0
        else:
            self.n_classes = n_classes
        self.reverse = reverse
        # self.depth_class_range = [6, 8, 10, 14, 18, 23, 31, 40, 49, 59, 92, 147, 236, 655]
        # self.n_classes = len(self.depth_class_range) + 1 # +1 including background

        self.eps = torch.tensor(0.1)
        self.eps = self.eps.to(device=self.device, dtype=torch.float32)
        self.loss_aux_past = None
        self.loss_main_past = None
        self.diff_aux = None
        self.loss_aux_current = None
        self.loss_main_current = None

        self.loss_aux_begin = None
        self.loss_main_begin = None

        self.pixel_num = None

        # Loss Weight Post processing. Using Adam optimizer
        alpha = torch.tensor(0.0002)
        beta1 = torch.tensor(0.9)
        beta2 = torch.tensor(0.999)
        eps = torch.tensor(1e-8)
        moment_vector1 = torch.zeros(self.n_classes - 1)
        moment_vector2 = torch.zeros(self.n_classes - 1)
        timestep = torch.tensor(1.0)

        self.alpha = alpha.to(device=device, dtype=torch.float32)
        self.beta1 = beta1.to(device=device, dtype=torch.float32)
        self.beta2 = beta2.to(device=device, dtype=torch.float32)
        self.eps = eps.to(device=device, dtype=torch.float32)
        self.moment_vector1 = moment_vector1.to(device=device, dtype=torch.float32)
        self.moment_vector2 = moment_vector2.to(device=device, dtype=torch.float32)
        self.timestep = timestep.to(device=device, dtype=torch.float32)

        return

    def get_aux_class_num(self):
        return self.n_classes - 1 # excluding ignore index

    def _get_depth_onehot_mask(self, true_deps, deps_shape):
        deps_mask_idx = np.zeros((true_deps.shape[0], 1, true_deps.shape[2], true_deps.shape[3]),
                                 np.int)  # deps_onehot_mask : (B, C, H, W)

        for dd in MultiTaskLossNYU.depth_class_range:
            t = (true_deps >= dd) * (true_deps != 0)
            deps_mask_idx += t.cpu().numpy()

        deps_onehot_mask = torch.zeros((deps_shape[0], self.n_classes-1, deps_shape[2], deps_shape[3]), dtype=torch.float32)
        deps_onehot_mask = deps_onehot_mask.to(device=self.device, dtype=torch.float32)
        deps_mask_idx = torch.LongTensor(deps_mask_idx).to(device=self.device, dtype=torch.long)
        deps_onehot_mask.scatter_(dim=1, index=deps_mask_idx, src=torch.tensor(1.0))
        return deps_onehot_mask.to(device=self.device, dtype=torch.float32)


    def _get_seg_onehot_mask(self, true_masks, masks_shape):
        # sum mask where y_i=c, instead of o_i=c
        masks_idx = torch.unsqueeze(true_masks, dim=1)  # (B, H, W) -> (B, 1, H, W)
        # masks_onehot = torch.FloatTensor(masks_shape)  # new tensor of shape (B, C, H, W)
        masks_onehot = torch.zeros((masks_shape), dtype=torch.float32)
        masks_onehot = masks_onehot.to(device=self.device, dtype=torch.float32)
        # masks_onehot.zero_()  # set all elements as 0
        masks_onehot.scatter_(dim=1, index=masks_idx,
                              src=torch.tensor(1.0))  # masks_onehot : (B, C, H, W)
        masks_onehot = masks_onehot[0][1:]
        masks_onehot = masks_onehot.reshape((-1, masks_onehot.shape[0], masks_onehot.shape[1], masks_onehot.shape[2]))

        return masks_onehot


    def forward(self, loss_weight, masks_pred, deps_pred, true_masks, true_deps):
        criterion_seg = nn.CrossEntropyLoss(reduction='none', ignore_index=0)
        criterion_dep = nn.L1Loss(reduction='none')

        deps_active_mask = (true_deps != 0)
        deps_active_pixel_num = torch.sum(deps_active_mask) #.detach().cpu()
        deps_pred = deps_active_mask.type(torch.float) * deps_pred

        seg_active_mask = (true_masks != 0)
        seg_active_pixel_num = torch.sum(seg_active_mask)

        loss_seg_mat = criterion_seg(masks_pred, true_masks)  # (B, H, W)
        loss_dep_mat = criterion_dep(deps_pred, true_deps)  # (B, 1, H, W)
        loss_dep_mat = torch.squeeze(loss_dep_mat, dim=1)  # (B, H, W)

        # ======= Only this part is varying by task =======
        if self.reverse:
            loss_aux_mat = loss_dep_mat
            loss_main_mat = loss_seg_mat
            aux_one_hot_of_c = self._get_depth_onehot_mask(true_deps, deps_pred.shape)
            # aux_one_hot_of_c = aux_one_hot_of_c.to(device=self.device, dtype=torch.float32)
            aux_pixel_num = deps_active_pixel_num
            main_pixel_num = seg_active_pixel_num
        else:
            loss_aux_mat = loss_seg_mat
            loss_main_mat = loss_dep_mat
            aux_one_hot_of_c = self._get_seg_onehot_mask(true_masks, masks_pred.shape)
            aux_pixel_num = seg_active_pixel_num
            main_pixel_num = deps_active_pixel_num
        # =================================================

        # List of pixel numbers in each class ex) pixel_num = torch.tensor([C1_pixel_num, C2_pixel_num, ....])
        # self.pixel_num  = torch.tensor([torch.sum(aux_one_hot_of_c[:,i]) for i in range(self.n_classes)])
        # self.pixel_num  = self.pixel_num .to(device=self.device, dtype=torch.float32)
        # self.pixel_num [self.pixel_num  == 0] = 1.0  # set 0.0 element as 1.0 to prevent 0/0= NaN
        self.pixel_num = true_masks.shape[1] * true_masks.shape[2]

        # List of dLs(c)/dW(c) ex) [dLs(c1)/dW(c1), dLs(c2)/dW(c2), ...]
        self.diff_aux = torch.tensor([torch.sum(aux_one_hot_of_c[:, i]*loss_aux_mat)
                                      for i in range(0, self.n_classes - 1)])
        # List of 1/N*(dLs(c)/dW(c))
        self.diff_aux = self.diff_aux / aux_pixel_num
        # List of Ld(c) ex) [Ld(c1), Ld(c2), Ld(c3), ...]
        self.loss_main_current = torch.tensor([torch.sum(aux_one_hot_of_c[:, i]*loss_main_mat)
                                              for i in range(0, self.n_classes - 1)])
        self.diff_aux = self.diff_aux.to(device=self.device, dtype=torch.float32)
        self.loss_main_current = self.loss_main_current.to(device=self.device, dtype=torch.float32)

        # loss_aux = torch.mean(self.loss_aux_current/self.pixel_num)

        loss_aux = list()
        for i in range(0, self.n_classes - 1):  # 0 is ignored index
            t = loss_weight[i] * torch.sum(aux_one_hot_of_c[:, i] * loss_aux_mat)
            loss_aux.append(t)
        loss_aux = torch.stack(loss_aux, 0) / aux_pixel_num

        # List of Ls(c) ex) [Ls(c1), Ls(c2), Ls(c3), ...]
        self.loss_aux_current = loss_aux.clone().detach()

        loss_aux = torch.sum(loss_aux)
        # loss_aux = torch.mean(criterion_seg(masks_pred, true_masks))

        #criterion_seg(masks_pred, true_masks)  #loss_seg = criterion_dep(masks_pred, true_masks) ver 2
        loss_main = torch.sum(loss_main_mat) / main_pixel_num

        if type(self.loss_aux_begin)==type(None):
            self.loss_aux_begin = loss_aux.clone().detach()
            self.loss_main_begin = loss_main.clone().detach()
            self.loss_aux_begin = self.loss_aux_begin.to(device=self.device, dtype=torch.float32)
            self.loss_main_begin = self.loss_main_begin.to(device=self.device, dtype=torch.float32)
        loss = loss_aux/self.loss_aux_begin + loss_main/self.loss_main_begin #0.810534*loss_seg/self.loss_seg_begin + loss_dep/self.loss_dep_begin
        return loss

    def loss_weight_update(self, loss_weight):

        if type(self.loss_aux_past) == type(None):
            self.loss_aux_past = self.loss_aux_current.clone().detach()
            self.loss_main_past = self.loss_main_current.clone().detach()
            self.loss_aux_past = self.loss_aux_past.to(device=self.device, dtype=torch.float32)
            self.loss_main_past = self.loss_main_past.to(device=self.device, dtype=torch.float32)
            return loss_weight
        else:
            # 1/N*{Ls(c)(t)/Ls(c)(0) - Ls(c)(t-1)/Ls(c)(0)}
            delta_aux = (self.loss_aux_current - self.loss_aux_past)/(self.loss_aux_begin)

            # 1/N*{Ld(c)(t)/Ld(c)(0) - Ld(c)(t-1)/Ld(c)(0)}
            delta_main = (self.loss_main_current - self.loss_main_past)/(self.loss_main_begin)

            # Update Ls(c)(t-1) and Ld(c)(t-1)
            self.loss_aux_past = self.loss_aux_current.clone().detach()
            self.loss_main_past = self.loss_main_current.clone().detach()
            self.loss_aux_past = self.loss_aux_past.to(device=self.device, dtype=torch.float32)
            self.loss_main_past = self.loss_main_past.to(device=self.device, dtype=torch.float32)

            loss_weight_update = ((delta_main+ self.eps) / (delta_aux + self.eps)) * self.diff_aux

            new_loss_weight = self.loss_weight_update_post_processing(loss_weight, loss_weight_update)

            return new_loss_weight

    def loss_weight_update_post_processing(self, loss_weight, loss_weight_update):

        # Code based on Adam algorithm
        moment_vector1 = self.beta1 * self.moment_vector1 + (1 - self.beta1) * loss_weight_update
        moment_vector2 = self.beta2 * self.moment_vector2 + (1 - self.beta2) * loss_weight_update ** 2

        m_hat = moment_vector1 / (1 - self.beta1 ** self.timestep)
        v_hat = moment_vector2 / (1 - self.beta2 ** self.timestep)
        self.timestep += 1

        loss_weight = loss_weight - self.alpha * m_hat / (torch.sqrt(v_hat) + self.eps)
        # loss_weight = loss_weight - alpha * loss_weight_update
        loss_weight = nn.ReLU()(loss_weight)
        # loss_weight = loss_weight*len(loss_weight_temp) / (torch.sum(loss_weight) + eps)  #  normalize weight
        return loss_weight



