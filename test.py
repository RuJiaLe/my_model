# from model.utils import block_aspp_moudle, CS_attention_module
# import torch
# import time
# from model.ConvGRU import ConvGRUCell


# def count_param(model):
#     param_count = 0
#     for param in model.parameters():
#         param_count += param.view(-1).size()[0]
#     return param_count


# y = 0
#
# aspp = block_aspp_moudle(in_dim=2048, out_dim=1024, output_stride=16)
# ATT = CS_attention_module(in_channels=1024)
#
# start = time.time()
#
# for i in range(10):
#     x = torch.rand(1, 1024, 128, 128)
#     # y = aspp(x)
#     y = ATT(x)
#
# end = time.time()
#
# print(y.size())
# print((end - start) / 10)
# print(f'aspp_parameter: {count_param(ATT)}')

# out1_1 = torch.rand(1, 1024, 8, 8)
# out1_2 = torch.rand(1, 1024, 8, 8)
# out1_3 = torch.rand(1, 1024, 8, 8)
# out1_4 = torch.rand(1, 1024, 8, 8)
#
# ConvGRU = ConvGRUCell(1024, 1024)
#
# start = time.time()
#
# for i in range(10):
#     out0_1 = ConvGRU(out1_1, None)
#     out0_2 = ConvGRU(out1_2, out0_1)
#     out0_3 = ConvGRU(out1_3, out0_2)
#     out0_4 = ConvGRU(out1_4, out0_3)
#
# end = time.time()
#
# print((end - start) / 10.0)
# print(f'GRU_parameter: {count_param(ConvGRU)}')


# import os
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import argparse
# from dataset.dataload import VideoDataset
# from dataset.transforms import get_train_transforms, get_transforms
# from torch.utils.data import DataLoader
# from model.train_model import Video_Encoder_Model, Video_Decoder_Model
# import torch.optim as optim
# from utils import adjust_lr, Eval_mae, Eval_F_measure, Eval_E_measure, Eval_S_measure
# from datetime import datetime
# from Loss import multi_bce_loss_fusion
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
# parser.add_argument('--save_model', type=str, default="./save_models/train_model", help='save_Encoder_model_path')
# parser.add_argument('--load_train_model', type=bool, default=False, help='load_model')
# parser.add_argument('--load_pre_train_model', type=bool, default=False, help='load_model')
# parser.add_argument('--dataset', type=list, default=["DAVIS", "DAVSOD", "UVSD"], help='dataset')
# parser.add_argument('--log_dir', type=str, default="./Log_file", help="log_dir file")
#
# args = parser.parse_args()
#
# # 数据加载
# # train data load
# train_transforms = get_train_transforms(input_size=(args.size, args.size))
# train_dataset = VideoDataset(root_dir="./data/train_data", training_set_list=args.dataset, training=True,
#                              transforms=train_transforms)
# train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=True)
#
# # val data load
# val_transforms = get_transforms(input_size=(args.size, args.size))
# val_dataset = VideoDataset(root_dir="./data/val_data", training_set_list=["DAVSOD"], training=True,
#                            transforms=val_transforms)
# val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=True)
#
# for i, packs in enumerate(train_dataloader):
#
#     images, gts = [], []
#     for pack in packs:
#         image, gt, path = pack["image"], pack["gt"], pack['path']
#         print(f'img_size: {image.size()}, gt_size: {gt.size()}, path: {path}')
