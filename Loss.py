from utils import SSIM, IOU
import torch.nn as nn

# Loss
bce_loss = nn.BCELoss(size_average=True)
ssim_loss = SSIM(window_size=11, size_average=True)
iou_loss = IOU(size_average=True)


def bce_ssim_loss(predict, target):
    bce_out = bce_loss(predict, target)
    ssim_out = 1 - ssim_loss(predict, target)
    iou_out = iou_loss(predict, target)

    loss = bce_out + ssim_out + iou_out

    return loss


def multi_bce_loss_fusion(frame1, frame2, frame3, frame4, gts):
    # 第一帧
    loss4_1 = bce_ssim_loss(frame1[0], gts[0])
    loss3_1 = bce_ssim_loss(frame2[0], gts[0])
    loss2_1 = bce_ssim_loss(frame3[0], gts[0])
    loss1_1 = bce_ssim_loss(frame4[0], gts[0])

    # 第二帧
    loss4_2 = bce_ssim_loss(frame1[1], gts[1])
    loss3_2 = bce_ssim_loss(frame2[1], gts[1])
    loss2_2 = bce_ssim_loss(frame3[1], gts[1])
    loss1_2 = bce_ssim_loss(frame4[1], gts[1])

    # 第一帧
    loss4_3 = bce_ssim_loss(frame1[2], gts[2])
    loss3_3 = bce_ssim_loss(frame2[2], gts[2])
    loss2_3 = bce_ssim_loss(frame3[2], gts[2])
    loss1_3 = bce_ssim_loss(frame4[2], gts[2])

    # 第一帧
    loss4_4 = bce_ssim_loss(frame1[3], gts[3])
    loss3_4 = bce_ssim_loss(frame2[3], gts[3])
    loss2_4 = bce_ssim_loss(frame3[3], gts[3])
    loss1_4 = bce_ssim_loss(frame4[3], gts[3])

    frame1_loss = loss4_1 + loss3_1 + loss2_1 + loss1_1
    frame2_loss = loss4_2 + loss3_2 + loss2_2 + loss1_2
    frame3_loss = loss4_3 + loss3_3 + loss2_3 + loss1_3
    frame4_loss = loss4_4 + loss3_4 + loss2_4 + loss1_4

    return frame1_loss, frame2_loss, frame3_loss, frame4_loss
