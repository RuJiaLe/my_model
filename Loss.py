import torch

from utils.Other_material import SSIM, IOU, S_Loss
import torch.nn as nn

# Loss
bce_loss = nn.BCELoss(size_average=True)
ssim_loss = SSIM(window_size=11, size_average=True)
iou_loss = IOU(size_average=True)
s_loss = S_Loss()


def Loss(predict, target):
    bce_out = bce_loss(predict, target)
    ssim_out = 1 - ssim_loss(predict, target)
    iou_out = iou_loss(predict, target)
    s_out = s_loss(predict, target)

    loss = bce_out + ssim_out + iou_out + s_out

    return loss


def multi_loss(out, gts):
    all_loss = []
    for j in range(len(gts)):

        frame_loss = torch.tensor(0.0)
        for i in range(len(out)):
            loss = Loss(out[i][j], gts[j])
            frame_loss += loss

        all_loss.append(frame_loss)

    return all_loss
