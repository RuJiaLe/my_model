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


def multi_bce_loss_fusion(out4, out3, out2, out1, out0, gts):
    # 第一帧
    loss4_1 = bce_ssim_loss(out4[0], gts[0])
    loss3_1 = bce_ssim_loss(out3[0], gts[0])
    loss2_1 = bce_ssim_loss(out2[0], gts[0])
    loss1_1 = bce_ssim_loss(out1[0], gts[0])
    loss0_1 = bce_ssim_loss(out0[0], gts[0])

    # 第二帧
    loss4_2 = bce_ssim_loss(out4[1], gts[1])
    loss3_2 = bce_ssim_loss(out3[1], gts[1])
    loss2_2 = bce_ssim_loss(out2[1], gts[1])
    loss1_2 = bce_ssim_loss(out1[1], gts[1])
    loss0_2 = bce_ssim_loss(out0[1], gts[1])

    # 第三帧
    loss4_3 = bce_ssim_loss(out4[2], gts[2])
    loss3_3 = bce_ssim_loss(out3[2], gts[2])
    loss2_3 = bce_ssim_loss(out2[2], gts[2])
    loss1_3 = bce_ssim_loss(out1[2], gts[2])
    loss0_3 = bce_ssim_loss(out0[2], gts[2])

    # 第四帧
    loss4_4 = bce_ssim_loss(out4[3], gts[3])
    loss3_4 = bce_ssim_loss(out3[3], gts[3])
    loss2_4 = bce_ssim_loss(out2[3], gts[3])
    loss1_4 = bce_ssim_loss(out1[3], gts[3])
    loss0_4 = bce_ssim_loss(out0[3], gts[3])

    frame1_loss = loss4_1 + loss3_1 + loss2_1 + loss1_1 + loss0_1
    frame2_loss = loss4_2 + loss3_2 + loss2_2 + loss1_2 + loss0_2
    frame3_loss = loss4_3 + loss3_3 + loss2_3 + loss1_3 + loss0_3
    frame4_loss = loss4_4 + loss3_4 + loss2_4 + loss1_4 + loss0_4

    return frame1_loss, frame2_loss, frame3_loss, frame4_loss
