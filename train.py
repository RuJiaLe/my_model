import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
from dataset.dataload import VideoDataset
from dataset.transforms import get_train_transforms, get_transforms
from torch.utils.data import DataLoader
from model.train_model import Video_Encoder_Model, Video_Decoder_Model
import torch.optim as optim
from utils import adjust_lr, Eval_mae, Eval_F_measure, Eval_E_measure, Eval_S_measure
from datetime import datetime
from Loss import multi_bce_loss_fusion
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
parser.add_argument('--save_model', type=str, default="./save_models/train_model", help='save_Encoder_model_path')
parser.add_argument('--load_train_model', type=bool, default=False, help='load_model')
parser.add_argument('--load_pre_train_model', type=bool, default=False, help='load_model')
parser.add_argument('--dataset', type=list, default=["DAVIS", "DAVSOD"], help='dataset')
parser.add_argument('--log_dir', type=str, default="./Log_file", help="log_dir file")

args = parser.parse_args()

# log
logging.basicConfig(filename=args.log_dir + '/train_log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

# 模型加载
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

Encoder_Model = Video_Encoder_Model(output_stride=16, input_channels=3, pretrained=True)
Decoder_Model = Video_Decoder_Model()

# 加载预训练模型
if args.load_pre_train_model:
    Encoder_path = "./save_models/pretrain_model/pre_encoder_model_15.pth"
    Decoder_path = "./save_models/pretrain_model/pre_decoder_model_15.pth"
    print('Loading state dict from: {}'.format(Encoder_path))

    Encoder_Model.load_state_dict(torch.load(Encoder_path, map_location=torch.device(device)))
    Decoder_dict = Decoder_Model.state_dict()
    pre_trained_dict = torch.load(Decoder_path, map_location=torch.device(device))

    for k, v in pre_trained_dict.items():
        if k in Decoder_dict:
            print("load:%s" % k)
    pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if (k in Decoder_dict)}
    Decoder_dict.update(pre_trained_dict)
    Decoder_Model.load_state_dict(Decoder_dict)
    print('Loading Decoder_Model state dick done!!!')

# 加载训练模型
if args.load_train_model:
    Encoder_path = args.save_model + "/encoder_model_15.pth"
    Decoder_path = args.save_model + "/decoder_model_15.pth"
    print('Loading state dict from: {}'.format(Encoder_path))
    Encoder_Model.load_state_dict(torch.load(Encoder_path, map_location=torch.device(device)))
    print('Loading state dict from: {}'.format(Decoder_path))
    Decoder_Model.load_state_dict(torch.load(Decoder_path, map_location=torch.device(device)))

if torch.cuda.is_available():
    Encoder_Model.cuda()
    Decoder_Model.cuda()

optimizer_Encoder = optim.Adam(Encoder_Model.parameters(), lr=args.lr)
optimizer_Decoder = optim.Adam(Decoder_Model.parameters(), lr=args.lr)


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
            torch.save(best_model['e_model'], args.save_model + '/best_encoder_model.pth')
            torch.save(best_model['d_model'], args.save_model + '/best_decoder_model.pth')
            logging.info('best_model_Epoch'.format(Epoch))
            print('find best model!!!')
            return True

    # 模型保存
    if Epoch % 10 == 0:
        torch.save(encoder_model.state_dict(),
                   args.save_model + '/encoder_model' + '_%d' % (Epoch + 0) + '.pth')
        torch.save(decoder_model.state_dict(),
                   args.save_model + '/decoder_model' + '_%d' % (Epoch + 0) + '.pth')


if __name__ == '__main__':
    print("start training!!!")
    # 数据加载
    # train data load
    train_transforms = get_train_transforms(input_size=(args.size, args.size))
    train_dataset = VideoDataset(root_dir="./train_data", training_set_list=args.dataset, training=True,
                                 transforms=train_transforms)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    # val data load
    val_transforms = get_transforms(input_size=(args.size, args.size))
    val_dataset = VideoDataset(root_dir="./val_data", training_set_list=["DAVSOD"], training=True,
                               transforms=val_transforms)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=True)

    best_model = {"loss": 0, 'e_model': Encoder_Model.state_dict(), 'd_model': Decoder_Model.state_dict()}
    flag = False

    for epoch in range(1, args.epoch + 1):
        adjust_lr(optimizer_Encoder, epoch, args.decay_rate, args.decay_epoch)
        adjust_lr(optimizer_Decoder, epoch, args.decay_rate, args.decay_epoch)

        flag = train(train_dataloader, val_dataloader, Encoder_Model, Decoder_Model, optimizer_Encoder, optimizer_Decoder, epoch)
        if flag:
            break

    print('-------------Congratulations! Training Done!!!-------------')
