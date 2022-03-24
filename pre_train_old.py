import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
from dataset.dataload import ImageDataset, VideoDataset
from dataset.transforms import get_train_transforms, get_transforms
from torch.utils.data import DataLoader
from model.pre_train_model import Video_Encoder_Model, Video_Decoder_Model
import torch.optim as optim
from utils import adjust_lr
from datetime import datetime
import utils
import logging

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=30, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--size', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=20, help='every n epochs decay learning rate')
parser.add_argument('--save_model', type=str, default="./save_models/pretrain_model", help='save_model')
parser.add_argument('--dataset', type=list, default=["DUTS"], help='dataset')
parser.add_argument('--log_dir', type=str, default="./Log_file", help="log_dir file")
parser.add_argument('--load_model', type=bool, default=False, help='load_model')

args = parser.parse_args()

# log
logging.basicConfig(filename=args.log_dir + '/pre_train_log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

# 数据加载
# train data load
pre_train_transforms = get_train_transforms(input_size=(args.size, args.size))
pre_train_dataset = ImageDataset(root_dir="./pre_train_data", training_set_list=args.dataset,
                                 image_transform=pre_train_transforms)
pre_train_dataloader = DataLoader(dataset=pre_train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

# 模型加载
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

Encoder_Model = Video_Encoder_Model(output_stride=16, input_channels=3, pretrained=True)
Decoder_Model = Video_Decoder_Model()

if args.load_model:
    Encoder_path = args.save_model + "/pre_encoder_model_15.pth"
    Decoder_path = args.save_model + "/pre_decoder_model_15.pth"
    print('Loading state dict from: {0}'.format(Encoder_path))
    Encoder_Model.load_state_dict(torch.load(Encoder_path, map_location=torch.device(device)))
    print('Loading state dict from: {0}'.format(Decoder_path))
    Decoder_Model.load_state_dict(torch.load(Decoder_path, map_location=torch.device(device)))

if torch.cuda.is_available():
    Encoder_Model.cuda()
    Decoder_Model.cuda()

optimizer_Encoder = optim.Adam(Encoder_Model.parameters(), lr=args.lr)
optimizer_Decoder = optim.Adam(Decoder_Model.parameters(), lr=args.lr)

# Loss
bce_loss = nn.BCELoss(size_average=True)
ssim_loss = utils.SSIM(window_size=11, size_average=True)
iou_loss = utils.IOU(size_average=True)


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


# 开始训练
def train(train_dataloader, Encoder_Model, Decoder_Model, optimizer_Encoder, optimizer_Decoder, epoch):
    Encoder_Model.train()
    Decoder_Model.train()

    total_step = len(pre_train_dataloader)
    losses = 0.0

    for i, packs in enumerate(train_dataloader):

        optimizer_Encoder.zero_grad()
        optimizer_Decoder.zero_grad()
        loss = 0.0
        blocks, gts = [], []
        for pack in packs:
            image, gt = pack["image"], pack["gt"]

            if torch.cuda.is_available():
                image, gt = Variable(image.cuda(), requires_grad=False), Variable(gt.cuda(), requires_grad=False)
            else:
                image, gt = Variable(image, requires_grad=False), Variable(gt, requires_grad=False)

            # 编码
            block = Encoder_Model(image)

            blocks.append(block)
            gts.append(gt)

        # 解码
        out1, out2, out3, out4 = Decoder_Model(blocks)

        # Loss 计算
        loss = 0.0
        frame1_loss, frame2_loss, frame3_loss, frame4_loss = multi_bce_loss_fusion(out1, out2, out3, out4, gts)
        loss = (frame1_loss + frame2_loss + frame3_loss + frame4_loss) / len(gts)
        losses += loss.data / 4.0

        # 反向传播
        loss.backward()
        optimizer_Encoder.step()
        optimizer_Decoder.step()

        # 显示内容
        if i % 1 == 0 or i == total_step - 1:
            print('{}, Epoch: [{:03d}/{:03d}], Step: [{:04d}/{:04d}], Loss: {:0.4f}'.format(datetime.now(),
                                                                                            epoch, args.epoch, i + 1,
                                                                                            total_step, losses / (i + 1)))

            logging.info('{}, Epoch: [{:03d}/{:03d}], Step: [{:04d}/{:04d}], Loss: {:0.4f}'.format(datetime.now(),
                                                                                                   epoch, args.epoch, i + 1,
                                                                                                   total_step, losses / (i + 1)))

    # 模型保存
    if epoch % 10 == 0:
        # 保存模型
        torch.save(Encoder_Model.state_dict(),
                   args.save_model + '/pre_encoder_model' + '_%d' % (epoch + 0) + '.pth')
        torch.save(Decoder_Model.state_dict(),
                   args.save_model + '/pre_decoder_model' + '_%d' % (epoch + 0) + '.pth')


if __name__ == '__main__':
    print("start training!!!")

    for epoch in range(1, args.epoch + 1):
        adjust_lr(optimizer_Encoder, epoch, args.decay_rate, args.decay_epoch)
        adjust_lr(optimizer_Decoder, epoch, args.decay_rate, args.decay_epoch)

        train(pre_train_dataloader, Encoder_Model, Decoder_Model, optimizer_Encoder, optimizer_Decoder, epoch)

    print('-------------Congratulations! Training Done!!!-------------')
