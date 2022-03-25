import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
from dataset.dataload import ImageDataset
from dataset.transforms import get_train_transforms, get_transforms
from torch.utils.data import DataLoader
from model.pre_train_model import Video_Encoder_Model, Video_Decoder_Model
import torch.optim as optim
from utils import adjust_lr, Eval_mae, Eval_F_measure, Eval_E_measure, Eval_S_measure
from datetime import datetime
import utils
import logging
import time

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=20, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--size', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=15, help='every n epochs decay learning rate')
parser.add_argument('--save_model', type=str, default="./save_models/pretrain_model", help='save_Encoder_model_path')
parser.add_argument('--load_model', type=bool, default=False, help='load_model')
parser.add_argument('--log_dir', type=str, default="./Log_file", help="log_dir file")

args = parser.parse_args()

# log
logging.basicConfig(filename=args.log_dir + '/pre_train_log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

# 模型加载
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

Encoder_Model = Video_Encoder_Model(output_stride=16, input_channels=3, pretrained=True)
Decoder_Model = Video_Decoder_Model()

# 加载训练模型
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


# val
def val(dataloader, val_encoder, val_decoder):
    val_encoder.eval()
    val_decoder.eval()

    total_num = len(dataloader) * 4
    MAES, E_measures, S_measures = 0.0, 0.0, 0.0
    img_num = 0
    avg_p, avg_r = 0.0, 0.0

    start_time = time.time()
    for i, packs in enumerate(dataloader):
        in_time = time.time()
        blocks, gts = [], []
        for pack in packs:
            img_num = img_num + 1
            image, gt = pack["image"], pack["gt"]

            if torch.cuda.is_available():
                image, gt = Variable(image.cuda(), requires_grad=False), Variable(gt.cuda(), requires_grad=False)
            else:
                image, gt = Variable(image, requires_grad=False), Variable(gt, requires_grad=False)

            # 编码
            block = val_encoder(image)

            blocks.append(block)
            gts.append(gt)

        # 解码
        out1, out2, out3, out4 = val_decoder(blocks)
        out_time = time.time()

        # Loss 计算
        predicts = [out1, out2, out3, out4]

        for predict, gt in (zip(predicts, gts)):
            mae = Eval_mae(predict, gt)
            MAES += mae

            prec, recall = Eval_F_measure(predict, gt)
            avg_p += prec
            avg_r += recall

            E_measure = Eval_E_measure(predict, gt)
            E_measures += E_measure
            S_measure = Eval_S_measure(predict, gt)
            S_measures += S_measure

            speed = out_time - in_time
            print('#val# {}, done: {:0.2f}%, img: {}/{}, mae: {:0.4f}, E_measure: {:0.4f}, S_measure: {:0.4f}, speed: {}'.
                  format(datetime.now().strftime('%m/%d %H:%M'), (img_num / total_num) * 100, img_num, total_num, mae, E_measure, S_measure, speed))
            logging.info('#val# {}, done: {:0.2f}%, img: {}/{}, mae: {:0.4f}, E_measure: {:0.4f}, S_measure: {:0.4f}, speed: {}'.
                         format(datetime.now().strftime('%m/%d %H:%M'), (img_num / total_num) * 100, img_num, total_num, mae, E_measure, S_measure, speed))

    avg_mae = MAES / img_num

    beta2 = 0.3
    avg_p = avg_p / img_num
    avg_r = avg_r / img_num
    score = (1 + beta2) * avg_p * avg_r / (beta2 * avg_p + avg_r)
    score[score != score] = 0
    max_F_measure = score.max()

    avg_E_measure = E_measures / img_num
    avg_S_measure = S_measures / img_num

    end_time = time.time()
    total_speed = end_time - start_time

    print('#val# {}, total_img: {}, avg_mae: {:0.4f}, max_F_measure: {:0.4f}, avg_E_measure: {:0.4f}, avg_S_measure: {:0.4f}, total_speed: {:0.4f}'.
          format(datetime.now().strftime('%m/%d %H:%M'), img_num, avg_mae, max_F_measure, avg_E_measure, avg_S_measure, total_speed))

    logging.info('{}'.format('*' * 100))
    logging.info('#val# {}, total_img: {} avg_mae: {:0.4f}, max_F_measure: {:0.4f}, avg_E_measure: {:0.4f}, avg_S_measure: {:0.4f}, total_speed: {:0.4f}'.
                 format(datetime.now().strftime('%m/%d %H:%M'), img_num, avg_mae, max_F_measure, avg_E_measure, avg_S_measure, total_speed))
    logging.info('{}'.format('*' * 100))

    total_loss = (1 - avg_mae) + max_F_measure + avg_S_measure + avg_S_measure

    return total_loss


# 开始训练
def train(train_data, val_data, encoder_model, decoder_model, optimizer_encoder, optimizer_decoder, Epoch):
    encoder_model.train()
    decoder_model.train()
    encoder_model.module.freeze_bn()
    decoder_model.module.freeze_bn()

    total_step = len(train_data)
    losses = []

    for i, packs in enumerate(train_data):
        i = i + 1

        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()

        blocks, gts = [], []
        for pack in packs:
            image, gt = pack["image"], pack["gt"]

            if torch.cuda.is_available():
                image, gt = Variable(image.cuda(), requires_grad=False), Variable(gt.cuda(), requires_grad=False)
            else:
                image, gt = Variable(image, requires_grad=False), Variable(gt, requires_grad=False)

            # 编码
            block = encoder_model(image)

            blocks.append(block)
            gts.append(gt)

        # 解码
        out1, out2, out3, out4 = decoder_model(blocks)

        # Loss 计算
        loss = []
        frame1_loss, frame2_loss, frame3_loss, frame4_loss = multi_bce_loss_fusion(out1, out2, out3, out4, gts)
        loss = [frame1_loss, frame2_loss, frame3_loss, frame4_loss]
        losses += loss

        # 反向传播
        torch.autograd.backward(loss)

        optimizer_encoder.step()
        optimizer_decoder.step()

        # 显示内容
        if i % 1 == 0 or i == total_step:
            print('#Train# {}, Epoch: [{}/{}], Done: {:0.2f}%, Step: [{}/{}], Loss: {:0.4f}'.
                  format(datetime.now().strftime('%m/%d %H:%M'),
                         Epoch, args.epoch, (i / total_step) * 100, i,
                         total_step, torch.mean(torch.tensor(losses))))

            logging.info('#Train# {}, Epoch: [{}/{}], Done: {:0.2f}%, Step: [{}/{}], Loss: {:0.4f}'.
                         format(datetime.now().strftime('%m/%d %H:%M'),
                                Epoch, args.epoch, (i / total_step) * 100, i,
                                total_step, torch.mean(torch.tensor(losses))))
    # 验证
    if Epoch % 2 == 0:
        val_loss = val(val_data, encoder_model, decoder_model)
        global best_model
        if best_model['loss'] < val_loss:
            best_model['loss'] = val_loss
            best_model['e_model'] = encoder_model.state_dict()
            best_model['d_model'] = decoder_model.state_dict()
        else:
            torch.save(best_model['e_model'], args.save_model + '/best_pre_encoder_model.pth')
            torch.save(best_model['d_model'], args.save_model + '/best_pre_decoder_model.pth')
            logging.info('best_model_Epoch'.format(Epoch))
            print('find best model!!!')
            return True

    # 模型保存
    if Epoch % 10 == 0:
        torch.save(encoder_model.state_dict(),
                   args.save_model + '/pre_encoder_model' + '_%d' % (Epoch + 0) + '.pth')
        torch.save(decoder_model.state_dict(),
                   args.save_model + '/pre_decoder_model' + '_%d' % (Epoch + 0) + '.pth')


if __name__ == '__main__':
    print("start training!!!")
    # 数据加载
    # train data load
    pre_train_transforms = get_train_transforms(input_size=(args.size, args.size))
    pre_train_dataset = ImageDataset(root_dir="./pre_train_data", training_set_list=['DUTS'],
                                     image_transform=pre_train_transforms)
    pre_train_dataloader = DataLoader(dataset=pre_train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    # val data load
    pre_val_transforms = get_transforms(input_size=(args.size, args.size))
    pre_val_dataset = ImageDataset(root_dir="./pre_val_data", training_set_list=["DUTS"],
                                   image_transform=pre_val_transforms)
    pre_val_dataloader = DataLoader(dataset=pre_val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=True)

    best_model = {"loss": 0, 'e_model': Encoder_Model.state_dict(), 'd_model': Decoder_Model.state_dict()}
    flag = False

    for epoch in range(1, args.epoch + 1):
        adjust_lr(optimizer_Encoder, epoch, args.decay_rate, args.decay_epoch)
        adjust_lr(optimizer_Decoder, epoch, args.decay_rate, args.decay_epoch)

        flag = train(pre_train_dataloader, pre_val_dataloader, Encoder_Model, Decoder_Model, optimizer_Encoder, optimizer_Decoder, epoch)
        if flag:
            break

    print('-------------Congratulations! Training Done!!!-------------')
