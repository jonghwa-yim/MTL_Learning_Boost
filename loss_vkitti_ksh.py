
import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(self, n_classes, device, reverse=False):
        """

        :rtype: nn.Module
        """
        super(MultiTaskLoss, self).__init__()
        self.device = device
        self.n_classes = n_classes
        self.eps = torch.tensor(0.1)
        self.eps = self.eps.to(device=self.device, dtype=torch.float32)
        self.loss_seg_past = None
        self.loss_dep_past = None
        self.diff_seg = None
        self.loss_seg_current = None
        self.loss_dep_current = None

        self.loss_seg_begin = None
        self.loss_dep_begin = None

        self.pixel_num = None

        # Loss Weight Post processing. Using Adam optimizer
        loss_weight = torch.ones(n_classes)
        alpha = torch.tensor(0.001)
        beta1 = torch.tensor(0.9)
        beta2 = torch.tensor(0.999)
        eps = torch.tensor(1e-8)
        moment_vector1 = torch.zeros(n_classes)
        moment_vector2 = torch.zeros(n_classes)
        timestep = torch.tensor(1.0)

        self.loss_weight = loss_weight.to(device=device, dtype=torch.float32)
        self.alpha = alpha.to(device=device, dtype=torch.float32)
        self.beta1 = beta1.to(device=device, dtype=torch.float32)
        self.beta2 = beta2.to(device=device, dtype=torch.float32)
        self.eps = eps.to(device=device, dtype=torch.float32)
        self.moment_vector1 = moment_vector1.to(device=device, dtype=torch.float32)
        self.moment_vector2 = moment_vector2.to(device=device, dtype=torch.float32)
        self.timestep = timestep.to(device=device, dtype=torch.float32)

        return

    def forward(self, loss_weight, masks_pred, deps_pred, true_masks, true_deps):
        criterion_seg = nn.CrossEntropyLoss(reduction='none')
        criterion_dep = nn.L1Loss(reduction='none')

        loss_seg_mat = criterion_seg(masks_pred, true_masks)  # (B, H, W)
        loss_dep_mat = criterion_dep(deps_pred, true_deps)  # (B, 1, H, W)
        loss_dep_mat = torch.squeeze(loss_dep_mat, dim=1)  # (B, H, W)

        masks_idx = torch.argmax(masks_pred, dim=1)  # masks_pred: (B, C, H, W), masks_idx: (B, H, W)
        masks_idx = torch.unsqueeze(masks_idx, dim=1) # (B, H, W) -> (B, 1, H, W)
        masks_onehot = torch.FloatTensor(masks_pred.shape)  # new tensor of shape (B, C, H, W)
        masks_onehot = masks_onehot.to(device=self.device, dtype=torch.float32)
        masks_onehot.zero_() # set all elements as 0
        masks_onehot.scatter_(dim=1, index=masks_idx, src=torch.tensor(1.0))  # masks_onehot : (B, C, H, W)

        # List of pixel numbers in each class ex) pixel_num = torch.tensor([C1_pixel_num, C2_pixel_num, ....])
        self.pixel_num  = torch.tensor([torch.sum(masks_onehot[:,i]) for i in range(self.n_classes)])
        self.pixel_num  = self.pixel_num .to(device=self.device, dtype=torch.float32)
        self.pixel_num [self.pixel_num  == 0] = 1.0  # set 0.0 element as 1.0 to prevent 0/0= NaN

        # List of dLs(c)/dW(c) ex) [dLs(c1)/dW(c1), dLs(c2)/dW(c2), ...]
        self.diff_seg = torch.tensor([torch.sum(masks_onehot[:, i]*loss_seg_mat)
                                      for i in range(self.n_classes)])
        # List of Ls(c) ex) [Ls(c1), Ls(c2), Ls(c3), ...]
        self.loss_seg_current = torch.tensor([loss_weight[i]*torch.sum(masks_onehot[:, i]*loss_seg_mat)
                                              for i in range(self.n_classes)])
        # List of Ld(c) ex) [Ld(c1), Ld(c2), Ld(c3), ...]
        self.loss_dep_current = torch.tensor([torch.sum(masks_onehot[:, i]*loss_dep_mat)
                                              for i in range(self.n_classes)])
        self.diff_seg = self.diff_seg.to(device=self.device, dtype=torch.float32)
        self.loss_seg_current = self.loss_seg_current.to(device=self.device, dtype=torch.float32)
        self.loss_dep_current = self.loss_dep_current.to(device=self.device, dtype=torch.float32)

        loss_seg = torch.mean(self.loss_seg_current/self.pixel_num)
        #criterion_seg(masks_pred, true_masks)  #loss_seg = criterion_dep(masks_pred, true_masks) ver 2
        loss_dep = torch.mean(loss_dep_mat)

        if type(self.loss_seg_begin)==type(None):
            self.loss_seg_begin = loss_seg.clone().detach()
            self.loss_dep_begin = loss_dep.clone().detach()
            self.loss_seg_begin = self.loss_seg_begin.to(device=self.device, dtype=torch.float32)
            self.loss_dep_begin = self.loss_dep_begin.to(device=self.device, dtype=torch.float32)
        loss = loss_seg/self.loss_seg_begin + loss_dep/self.loss_dep_begin #0.810534*loss_seg/self.loss_seg_begin + loss_dep/self.loss_dep_begin
        return loss

    def loss_weight_update(self, loss_weight):

        if type(self.loss_seg_past) == type(None):
            self.loss_seg_past = self.loss_seg_current.clone().detach()
            self.loss_dep_past = self.loss_dep_current.clone().detach()
            self.loss_seg_past = self.loss_seg_past.to(device=self.device, dtype=torch.float32)
            self.loss_dep_past = self.loss_dep_past.to(device=self.device, dtype=torch.float32)
            return loss_weight
        else:
            # List of 1/N*(dLs(c)/dW(c))
            self.diff_seg = self.diff_seg / self.pixel_num

            # 1/N*{Ls(c)(t)/Ls(c)(0) - Ls(c)(t-1)/Ls(c)(0)}
            delta_seg = (self.loss_seg_current - self.loss_seg_past)/(self.loss_seg_begin*self.pixel_num)

            # 1/N*{Ld(c)(t)/Ld(c)(0) - Ld(c)(t-1)/Ld(c)(0)}
            delta_dep = (self.loss_dep_current - self.loss_dep_past)/(self.loss_dep_begin*self.pixel_num)

            # Update Ls(c)(t-1) and Ld(c)(t-1)
            self.loss_seg_past = self.loss_seg_current.clone().detach()
            self.loss_dep_past = self.loss_dep_current.clone().detach()
            self.loss_seg_past = self.loss_seg_past.to(device=self.device, dtype=torch.float32)
            self.loss_dep_past = self.loss_dep_past.to(device=self.device, dtype=torch.float32)

            loss_weight_update = ((delta_dep+ self.eps) / (delta_seg + self.eps)) * self.diff_seg

            new_loss_weight = self.loss_weight_update_post_processing(loss_weight, loss_weight_update, delta_dep, delta_seg, self.diff_seg)

            return new_loss_weight

    def loss_weight_update_post_processing(self, loss_weight, loss_weight_update, delta_Ldep, delta_Lseg, diff_Lseg):

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

