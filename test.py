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


import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
from dataset.dataload import VideoDataset, ImageDataset
from dataset.transforms import get_train_transforms, get_transforms
from torch.utils.data import DataLoader
from model.video_train_model import Video_Encoder_Model, Video_Decoder_Model
import torch.optim as optim
from datetime import datetime
from Loss import multi_bce_loss_fusion
import logging
import time

parser = argparse.ArgumentParser()

# parameters
parser.add_argument('--epoch', type=int, default=60, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--size', type=int, default=256, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=15, help='every n epochs decay learning rate')
parser.add_argument('--log_dir', type=str, default="./Log_file", help="log_dir file")

# data_path
parser.add_argument('--image_train_path', type=str, default="./data/Image_train_data", help='image_train_path')
parser.add_argument('--image_val_path', type=str, default="./data/Image_val_data", help='image_val_path')

parser.add_argument('--video_train_path', type=str, default="./data/Video_train_data", help='video_train_path')
parser.add_argument('--video_val_path', type=str, default="./data/Video_val_data", help='video_val_path')

parser.add_argument('--predict_data_path', type=str, default="./data/Video_test_data", help='predict_data_path')

# dataset
parser.add_argument('--image_train_dataset', type=list, default=["DUTS"], help='image_train_dataset')
parser.add_argument('--image_val_dataset', type=list, default=["DUTS"], help='image_val_dataset')

parser.add_argument('--video_train_dataset', type=list, default=["DUTS"], help='video_train_dataset')
parser.add_argument('--video_val_dataset', type=list, default=["Val_data"], help='video_val_dataset')

parser.add_argument('--predict_dataset', type=str, default="DAVIS_20",
                    choices=["DAVIS_20", "DAVSOD_Difficult_15", "DAVSOD_Easy_25", "DAVSOD_Normal_15", "DAVSOD_Validation_Set_21", "UVSD_9"],
                    help='predict_dataset')

# model_path
parser.add_argument('--image_encoder_model', type=str, default="./save_models/image_train_model/best_image_encoder_model.pth", help='image_encoder_model')
parser.add_argument('--image_decoder_model', type=str, default="./save_models/image_train_model/best_image_decoder_model.pth", help='image_decoder_model')

parser.add_argument('--video_encoder_model', type=str, default="./save_models/video_train_model/best_video_encoder_model.pth", help='video_encoder_model')
parser.add_argument('--video_decoder_model', type=str, default="./save_models/video_train_model/best_video_decoder_model.pth", help='video_decoder_model')

args = parser.parse_args()

# 数据加载
# train data load
image_train_transforms = get_train_transforms(input_size=(args.size, args.size))
image_train_dataset = ImageDataset(root_dir=args.image_train_path, training_set_list=args.image_train_dataset,
                                   image_transform=image_train_transforms)
image_train_dataloader = DataLoader(dataset=image_train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

# val data load
image_val_transforms = get_transforms(input_size=(args.size, args.size))
image_val_dataset = ImageDataset(root_dir=args.image_val_path, training_set_list=args.image_val_dataset,
                                 image_transform=image_val_transforms)
image_val_dataloader = DataLoader(dataset=image_val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=True)

print(f'load image train data done, total train number is {len(image_train_dataloader) * 4}')
print(f'load image val data done, total train number is {len(image_val_dataloader) * 4}')

if __name__ == '__main__':
    for i, packs in enumerate(image_val_dataloader):

        images, gts = [], []
        for pack in packs:
            image, gt, path = pack["image"], pack["gt"], pack['path']
            print(f'img_size: {image.size()}, gt_size: {gt.size()}, path: {path}')
