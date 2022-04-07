import os

import torch
import time
from torch.autograd import Variable
from .Other_material import Eval_mae, Eval_F_measure, Eval_S_measure
import logging
from datetime import datetime
from Loss import multi_bce_loss_fusion
import torch.optim as optim
from .Other_material import adjust_lr

# best_val_loss
best_val_loss = 0.0


# val
def val(val_dataloader, model, batch_size):
    model.eval()

    total_num = len(val_dataloader) * 4 * batch_size
    MAES, S_measures = 0.0, 0.0
    img_num = 0
    avg_p, avg_r = 0.0, 0.0

    start_time = time.time()
    for i, packs in enumerate(val_dataloader):
        images, gts = [], []
        for pack in packs:
            image, gt = pack["image"], pack["gt"]

            if torch.cuda.is_available():
                image, gt = Variable(image.cuda(), requires_grad=False), Variable(gt.cuda(), requires_grad=False)
            else:
                image, gt = Variable(image, requires_grad=False), Variable(gt, requires_grad=False)

            images.append(image)
            gts.append(gt)

        # 解码
        out4, out3, out2, ou1, out0 = model(images)

        # Loss 计算
        predicts = out0

        for predict_, gt_ in (zip(predicts, gts)):
            for k in range(batch_size):
                predict = predict_[k, :, :, :].unsqueeze(0)
                gt = gt_[k, :, :, :].unsqueeze(0)

                img_num = img_num + 1
                mae = Eval_mae(predict, gt)
                MAES += mae.data

                prec, recall = Eval_F_measure(predict, gt)
                avg_p += prec.data
                avg_r += recall.data

                S_measure = Eval_S_measure(predict, gt)
                S_measures += S_measure.data

        if img_num % 200 == 0:
            print('#val# Done: {:0.2f}%, img: {}/{}, mae: {:0.4f}, S_measure: {:0.4f}'.
                  format((img_num / total_num) * 100, img_num, total_num, MAES / img_num, S_measures / img_num))

    avg_mae = MAES / img_num

    beta2 = 0.3
    avg_p = avg_p / img_num
    avg_r = avg_r / img_num
    score = (1 + beta2) * avg_p * avg_r / (beta2 * avg_p + avg_r)
    score[score != score] = 0
    max_F_measure = score.max()

    avg_S_measure = S_measures / img_num

    end_time = time.time()
    total_speed = end_time - start_time

    total_loss = (1 - avg_mae) + max_F_measure + avg_S_measure

    print('*' * 50)
    print('#val# {}, total_img: {}, avg_mae: {:0.4f}, max_F_measure: {:0.4f}, avg_S_measure: {:0.4f}, total_speed: {:0.4f}'.
          format(datetime.now().strftime('%m/%d %H:%M'), img_num, avg_mae, max_F_measure, avg_S_measure, total_speed))
    print('#val# total_loss: {:0.4f}'.format(total_loss))
    print('*' * 50)

    logging.info('{}'.format('*' * 100))
    logging.info('#val# {}, total_img: {} avg_mae: {:0.4f}, max_F_measure: {:0.4f}, avg_S_measure: {:0.4f}, total_speed: {:0.4f}'.
                 format(datetime.now().strftime('%m/%d %H:%M'), img_num, avg_mae, max_F_measure, avg_S_measure, total_speed))
    logging.info('#val# total_loss: {:0.4f}'.format(total_loss))
    logging.info('{}'.format('*' * 100))

    return total_loss


# 开始训练
def train(train_data, val_data, model, optimizer, Epoch, total_epoch, model_path, batch_size):
    model.train()
    model.freeze_bn()

    total_step = len(train_data)
    losses = 0.0

    for i, packs in enumerate(train_data):
        i = i + 1

        optimizer.zero_grad()

        images, gts = [], []
        for pack in packs:
            image, gt = pack["image"], pack["gt"]

            if torch.cuda.is_available():
                image, gt = Variable(image.cuda(), requires_grad=False), Variable(gt.cuda(), requires_grad=False)
            else:
                image, gt = Variable(image, requires_grad=False), Variable(gt, requires_grad=False)

            images.append(image)
            gts.append(gt)

        # 解码
        out4, out3, out2, out1, out0 = model(images)  # 第4阶段, 第3阶段, 第2阶段, 第1阶段, 第0阶段

        # Loss 计算
        loss = []
        frame1_loss, frame2_loss, frame3_loss, frame4_loss = multi_bce_loss_fusion(out4, out3, out2, out1, out0, gts)
        loss = [frame1_loss, frame2_loss, frame3_loss, frame4_loss]
        losses = losses + frame1_loss.data + frame2_loss.data + frame3_loss.data + frame4_loss.data

        # 反向传播
        torch.autograd.backward(loss)

        optimizer.step()

        # 显示与记录内容
        if i % 10 == 0 or i == total_step:
            print('#Train# {}, Epoch: [{}/{}], Done: {:0.2f}%, Step: [{}/{}], Loss: {:0.4f}'.
                  format(datetime.now().strftime('%m/%d %H:%M'),
                         Epoch, total_epoch, (i / total_step) * 100, i,
                         total_step, losses / (i * 4)))

        if i % 100 == 0 or i == total_step:
            logging.info('#Train# {}, Epoch: [{}/{}], Done: {:0.2f}%, Step: [{}/{}], Loss: {:0.4f}'.
                         format(datetime.now().strftime('%m/%d %H:%M'),
                                Epoch, total_epoch, (i / total_step) * 100, i,
                                total_step, losses / (i * 4)))

    # 验证与模型保存
    if Epoch % 1 == 0:
        val_loss = val(val_data, model, batch_size)
        global best_val_loss

        if best_val_loss < val_loss:  # big is best
            best_val_loss = val_loss

            torch.save({'epoch': Epoch, 'state_dict': model.state_dict(), 'best_val_loss': best_val_loss, 'optimizer': optimizer.state_dict()},
                       model_path)

            print('this is best_model_Epoch: {}'.format(Epoch))
            logging.info('this is best_model_Epoch: {}'.format(Epoch))


# 模型加载模式
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def start_train(train_dataloader, val_dataloader, model, total_epoch,
                lr, decay_rate, decay_epoch, model_path, log_dir, start_epoch, batch_size):
    # log
    logging.basicConfig(filename=log_dir + '/image_train_log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # 加载至cuda
    if torch.cuda.is_available():
        model.cuda()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 加载模型
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        start_epoch = checkpoint['epoch']

        global best_val_loss
        best_val_loss = checkpoint['best_val_loss']
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

    for epoch in range(start_epoch, total_epoch + 1):
        adjust_lr(optimizer, epoch, decay_rate, decay_epoch)

        train(train_dataloader, val_dataloader, model, optimizer, epoch, total_epoch, model_path, batch_size)
