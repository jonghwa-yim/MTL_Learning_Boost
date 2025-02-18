import os
import argparse
import torch
from torch.utils.data import DataLoader
from loss_vkitti import MultiTaskLossVkitti
from loss_nyuv2 import MultiTaskLossNYU
from loss_cityscapes import MultiTaskLossCity
from architecture.unet.unet_model import UNet
from dataset import vkitti, nyuv2, cityscapes


def argument_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='Multi-task Learning, the method of boosting the main task.')
    parser.add_argument('--dataset', type=str, help='The name of dataset to train and test',
                        choices=['vkitti', 'nyuv2', 'cityscapes'])
    parser.add_argument('--model', type=str, help='The model name of Neural Network',
                        choices=['unet', ])
    parser.add_argument('--reverse', dest='reverse', action='store_true',
                        help='Reverse main/aux tasks. If true, main:seg, aux:depth')
    parser.add_argument('--outputDir', default='', type=str, metavar='PATH',
                        help='Path to save trained model.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
    parser.add_argument('--nEpochs', type=int, default=5, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
    parser.add_argument('--lr_decay_ep', type=int, default=0, help='Epoch where the learning rate decay.')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--input_size', default=256, type=int, help='input image size. ')

    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')

    args = parser.parse_args()
    return args


def default_config(args):
    configs = dict()
    if args.dataset == 'vkitti':
        configs['dir_img'] = 'data/imgs/'
        configs['dir_mask'] = 'data/masks/'
        configs['dir_dep'] = 'data/depths/'
        configs['n_classes'] = 13
    elif args.dataset == 'nyuv2':
        configs['dir_img'] = 'data/imgs/'
        configs['dir_mask'] = 'data/label/'
        configs['dir_dep'] = 'data/depth/'
        configs['n_classes'] = 14
    elif args.dataset == 'cityscapes':
        configs['dir_img'] = 'img/'
        configs['dir_mask'] = 'seg/'
        configs['dir_dep'] = 'depth/'
        configs['n_classes'] = 34

    return configs


def train(epoch, loss_weight):
    model.train()
    epoch_loss = 0
    inter_loss = 0
    cur_stage = 0
    cnt = 0

    for iteration, batch in enumerate(train_loader):
        imgs = batch['image'].cuda(args.gpu, non_blocking=False)
        true_masks = batch['mask'].cuda(args.gpu, non_blocking=True)
        true_deps = batch['depth'].cuda(args.gpu, non_blocking=True)

        optimizer.zero_grad()

        output = model(imgs)
        masks_pred = output['segmentation_prediction']
        deps_pred = output['depth_prediction']

        loss = criterion(loss_weight, masks_pred, deps_pred, true_masks, true_deps)

        if False:
            loss_weight = criterion.loss_weight_update(loss_weight)

        epoch_loss += loss.item()
        inter_loss += loss.item()
        cnt += 1

        loss.backward()
        optimizer.step()

        if False: #iteration >= int(cur_stage  / 5 * len(train_loader)):
            print("==> Epoch[{}]({}%): ".format(epoch, int(cur_stage * 20)),
                  "Inter. Avg. Loss: {:.4f}".format(inter_loss / cnt))
            inter_loss = 0
            cnt = 0
            cur_stage += 1

    print("==> Epoch {} Complete: ".format(epoch),
          "Avg. Loss: {:.4f}".format(epoch_loss / len(train_loader)))
    return loss_weight


def accuracy_vkitti(masks_pred, deps_pred, true_masks, true_deps):
    import torch.nn.functional as F
    smooth = 1e-6
    seg_loss = 0
    dep_l1_loss = 0
    dep_rmse = 0
    seg_iou = 0

    masks_idx_pred = torch.argmax(masks_pred, dim=1)  # masks_pred: (B, C, H, W), masks_idx_pred: (B, H, W)
    masks_idx_pred = torch.unsqueeze(masks_idx_pred, dim=1)  # (B, H, W) -> (B, 1, H, W)
    masks_onehot_pred = torch.FloatTensor(masks_pred.shape)  # new tensor of shape (B, C, H, W)
    masks_onehot_pred = masks_onehot_pred.to(device=device, dtype=torch.long)
    masks_onehot_pred.zero_()  # set all elements as 0
    masks_onehot_pred.scatter_(dim=1, index=masks_idx_pred,
                               src=torch.tensor(1.0))  # masks_onehot_pred : (B, C, H, W)

    masks_idx_true = torch.unsqueeze(true_masks, dim=1)  # (B, H, W) -> (B, 1, H, W)
    masks_onehot_true = torch.FloatTensor(masks_pred.shape)  # new tensor of shape (B, C, H, W)
    masks_onehot_true = masks_onehot_true.to(device=device, dtype=torch.long)
    masks_onehot_true.zero_()  # set all elements as 0
    masks_onehot_true.scatter_(dim=1, index=masks_idx_true,
                               src=torch.tensor(1.0))  # masks_onehot_true : (B, C, H, W)

    intersection = (masks_onehot_pred & masks_onehot_true).float().sum(
        (0, 2, 3))  # will be 1 if pred=1 and true=1 intersection: (C)
    union = (masks_onehot_pred | masks_onehot_true).float().sum(
        (0, 2, 3))  # will be 1 if pred=1 or true=1 union: (C)

    iou = (intersection + smooth) / (union + smooth)  # smooth out devision to avoid 0/0
    seg_iou += iou.mean().item()

    for true_masks, true_deps, masks_pred, deps_pred in zip(true_masks, true_deps, masks_pred, deps_pred):
        seg_loss += F.cross_entropy(masks_pred.unsqueeze(dim=0), true_masks.unsqueeze(
            dim=0)).item()  # add segmentation loss
        dep_l1_loss += F.l1_loss(deps_pred, true_deps).item()
        dep_rmse += torch.sqrt(F.mse_loss(deps_pred, true_deps)).item()

    return seg_loss, seg_iou, dep_l1_loss, dep_rmse


def accuracy_nyuv2(masks_pred, deps_pred, true_masks, true_deps):
    import torch.nn
    import torch.nn.functional as F
    seg_loss = 0
    dep_l1_loss = 0
    dep_rmse = 0

    criterion_seg = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=0)

    seg_active_mask = (true_masks != 0)
    seg_active_pixel_num = torch.sum(seg_active_mask)

    loss_seg_mat = criterion_seg(masks_pred, true_masks)  # (B, H, W)

    seg_loss_t = torch.sum(loss_seg_mat) / seg_active_pixel_num
    seg_loss += seg_loss_t.item()

    criterion_dep = torch.nn.L1Loss(reduction='none')

    deps_active_mask = (true_deps != 0)
    deps_active_pixel_num = torch.sum(deps_active_mask)  # .detach().cpu()
    deps_pred = deps_active_mask.type(torch.float) * deps_pred

    loss_dep_mat = criterion_dep(deps_pred, true_deps)  # (B, 1, H, W)
    loss_dep_mat = torch.squeeze(loss_dep_mat, dim=1)  # (B, H, W)

    dep_l1_loss_t = torch.sum(loss_dep_mat) / deps_active_pixel_num
    dep_l1_loss += dep_l1_loss_t.item()

    dep_rmse_t = F.mse_loss(deps_pred, true_deps, reduction='sum') / deps_active_pixel_num
    dep_rmse += torch.sqrt(dep_rmse_t).item()

    return seg_loss, 0, dep_l1_loss, dep_rmse


def test(loss_weight):
    model.eval()
    tot_seg = 0
    tot_dep_l1 = 0
    tot_dep_rmse = 0
    tot_iou = 0

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['image'].cuda(args.gpu, non_blocking=False)
            true_masks = batch['mask'].cuda(args.gpu, non_blocking=True)
            true_deps = batch['depth'].cuda(args.gpu, non_blocking=True)

            output = model(imgs)
            masks_pred = output['segmentation_prediction']
            deps_pred = output['depth_prediction']

            if args.dataset == 'vkitti':
                acc_ret = accuracy_vkitti(masks_pred, deps_pred, true_masks, true_deps)
                tot_seg += acc_ret[0]
                tot_iou += acc_ret[1]
                tot_dep_l1 += acc_ret[2]
                tot_dep_rmse += acc_ret[3]
            elif args.dataset == 'nyuv2' or args.dataset == 'cityscapes':
                acc_ret = accuracy_nyuv2(masks_pred, deps_pred, true_masks, true_deps)
                tot_seg += acc_ret[0]
                tot_iou += acc_ret[1]
                tot_dep_l1 += acc_ret[2]
                tot_dep_rmse += acc_ret[3]

    print("===> Avg. Seg Loss (cross entropy): {:.6f} (px)".format(tot_seg / len(test_loader)))
    print("===> Avg. Seg IOU: {:.3f} (%)".format(tot_iou * 100 / len(test_loader)))
    print("===> Avg. L1Loss inverse depth: {:.6f} (m^-1)".format(tot_dep_l1 / len(test_loader)))
    print("===> Avg. RMSE inverse depth: {:.5f} (m^-1)".format(tot_dep_rmse / len(test_loader)))
    print("===> Classwise Weights: ", loss_weight.cpu(), "\n")
    return


def save_checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, os.path.join(args.outputDir, model_out_path))
    print("Checkpoint saved to {}\n".format(model_out_path))
    return



def adjust_learning_rate(optimizer, epoch, epoch_ch=4, lr_update=0.2):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (lr_update ** (epoch // epoch_ch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__ == '__main__':
    args = argument_parser()
    configs = default_config(args)
    print(args)

    device = torch.device('cuda:' + str(args.gpu))

    if not os.path.exists(args.outputDir):
        os.mkdir(args.outputDir)

    print('===> Loading datasets')
    if args.dataset == 'vkitti':
        # dataset = vkitti_numpy.BasicDataset(configs['dir_img'], configs['dir_mask'], configs['dir_dep'], args.scale)
        # val_percent = 0.0
        # test_percent = 0.1
        # n_val = int(len(dataset) * val_percent)
        # n_test = int(len(dataset) * test_percent)
        # n_train = len(dataset) - n_val - n_test
        # dataset_train, dataset_val, dataset_test = random_split(dataset, [n_train, n_val, n_test])
        dataset_train = vkitti.DatasetFromTXT('dataset/train_vkitti.txt')
        dataset_test = vkitti.DatasetFromTXT('dataset/test_vkitti.txt')
    elif args.dataset == 'nyuv2':
        dataset_train = nyuv2.NYUV2Dataset(trainval='train', img_size=256)
        dataset_test = nyuv2.NYUV2Dataset(trainval='val', img_size=256)
    elif args.dataset == 'cityscapes':
        dataset_train = cityscapes.CITISCAPESDataset(phase='train')
        dataset_test = cityscapes.CITISCAPESDataset(phase='val')
    else:
        dataset_train = None
        dataset_test = None

    train_loader = DataLoader(dataset_train, batch_size=args.batchSize, shuffle=True, num_workers=8, pin_memory=True)
    # val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=8, pin_memory=True) # batch_size=1 to mitigate memory load
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    print('===> Building model')
    model = None
    if args.model == 'unet':
        # n_channels=3 for RGB images
        # n_classes is the number of probabilities you want to get per pixel
        model = UNet(n_channels=3, n_classes=configs['n_classes']).cuda(args.gpu)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            model.load_state_dict(checkpoint.state_dict(), strict=True)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
    if args.dataset == 'vkitti':
        criterion = MultiTaskLossVkitti(model.n_classes, device, args.reverse)
    elif args.dataset == 'nyuv2':
        criterion = MultiTaskLossNYU(model.n_classes, device, args.reverse)
    elif args.dataset == 'cityscapes':
        criterion = MultiTaskLossCity(model.n_classes, device, args.reverse)

    print('===> Training model')
    epoch_ch = args.lr_decay_ep
    lr = args.lr

    # initialization of class-wise weights
    # if args.reverse:
    if args.dataset == 'vkitti':
        loss_weight = torch.ones(criterion.get_aux_class_num(), dtype=torch.float32).cuda(args.gpu)
    elif args.dataset == 'nyuv2' or args.dataset == 'cityscapes':
        loss_weight = torch.ones(criterion.get_aux_class_num(), dtype=torch.float32).cuda(args.gpu)
    # else:
    #     if args.dataset == 'vkitti':
    #         loss_weight = torch.ones(model.n_classes, dtype=torch.float32).cuda(args.gpu)
    #     elif args.dataset == 'nyuv2' or args.dataset == 'cityscapes':
    #         loss_weight = torch.ones(model.n_classes - 1, dtype=torch.float32).cuda(args.gpu)

    for epoch in range(1, args.nEpochs + 1):
        loss_weight = train(epoch, loss_weight)
        test(loss_weight)
        # if epoch == args.nEpochs or epoch % epoch_ch == 0 or epoch % int(epoch_ch/2):
        if epoch == args.nEpochs or epoch % 4 == 0:
            save_checkpoint(epoch)

        if epoch == args.lr_decay_ep:
            adjust_learning_rate(optimizer, epoch, epoch_ch, 0.1)

    print('Done')
