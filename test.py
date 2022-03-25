# import os
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import argparse
# from dataset.dataload import ImageDataset
# from dataset.transforms import get_train_transforms, get_transforms
# from torch.utils.data import DataLoader
# from model.pre_train_model import Video_Encoder_Model, Video_Decoder_Model
# import torch.optim as optim
# from utils import adjust_lr, Eval_mae, Eval_F_measure, Eval_E_measure, Eval_S_measure
# from datetime import datetime
# import utils
# import logging
# import time
#
# parser = argparse.ArgumentParser()
#
# parser.add_argument('--epoch', type=int, default=20, help='epoch number')
# parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
# parser.add_argument('--size', type=int, default=256, help='training dataset size')
# parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
# parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
# parser.add_argument('--decay_epoch', type=int, default=15, help='every n epochs decay learning rate')
# parser.add_argument('--save_model', type=str, default="./save_models/pretrain_model", help='save_Encoder_model_path')
# parser.add_argument('--load_model', type=bool, default=False, help='load_model')
# parser.add_argument('--log_dir', type=str, default="./Log_file", help="log_dir file")
#
# args = parser.parse_args()
#
# # train data load
# pre_train_transforms = get_train_transforms(input_size=(args.size, args.size))
# pre_train_dataset = ImageDataset(root_dir="./pre_train_data", training_set_list=['DUTS'],
#                                  image_transform=pre_train_transforms)
# pre_train_dataloader = DataLoader(dataset=pre_train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)
#
# # val data load
# pre_val_transforms = get_transforms(input_size=(args.size, args.size))
# pre_val_dataset = ImageDataset(root_dir="./pre_val_data", training_set_list=["DUTS"],
#                                image_transform=pre_val_transforms)
# pre_val_dataloader = DataLoader(dataset=pre_val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=True)
#
# if __name__ == '__main__':
#     for i, packs in enumerate(pre_val_dataloader):
#         for pack in packs:
#             image, gt = pack["image"], pack["gt"]
#             print(i, len(pre_val_dataloader) * 4, image.size(), gt.size())

import torch

x1 = torch.rand(1, 3, 256, 256)
x2 = torch.rand(1, 3, 256, 256)
x3 = torch.rand(1, 3, 256, 256)
x4 = torch.rand(1, 3, 256, 256)

# x = [x1, x2, x3, x4]
#
# y = torch.cat(x, dim=1)
# print(y.size())
# N, C, H, W = x1.size()
# print(x1.size()[1])

x = torch.rand(2, 3)
y = torch.rand(2, 3)
print(x, y)
print(x * y)
