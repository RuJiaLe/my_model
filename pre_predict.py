import os
import torch
from torch.autograd import Variable
import argparse
from dataset.dataload import VideoDataset
from dataset.transforms import get_transforms
from torch.utils.data import DataLoader
from model.pre_train_model import Video_Encoder_Model, Video_Decoder_Model
from utils import Save_result
import time

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--size', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--save_model', type=str, default="./save_models/pretrain_model", help='save_model')
parser.add_argument('--predict_data_path', type=str, default="./test_predict_data", help='predict_data_path')
parser.add_argument('--dataset', type=list, default=["DAVIS", "DAVSOD", "UVSD"], help='dataset')

args = parser.parse_args()

# 数据加载
transforms = get_transforms(input_size=(args.size, args.size))
dataset = VideoDataset(root_dir=args.predict_data_path, training_set_list=args.dataset, training=True,
                       transforms=transforms)
train_dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=True)

# 模型加载
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

Encoder_Model = Video_Encoder_Model(output_stride=16, input_channels=3, pretrained=False)
Decoder_Model = Video_Decoder_Model()

Encoder_path = args.save_model + "/pre_encoder_model_30.pth"
Decoder_path = args.save_model + "/pre_decoder_model_30.pth"
print('Loading state dict from: {0}'.format(Encoder_path))
Encoder_Model.load_state_dict(torch.load(Encoder_path, map_location=torch.device(device)))
print('Loading state dict from: {0}'.format(Decoder_path))
Decoder_Model.load_state_dict(torch.load(Decoder_path, map_location=torch.device(device)))

if torch.cuda.is_available():
    Encoder_Model.cuda()
    Decoder_Model.cuda()


# 开始predict
def predict(train_dataloader, Encoder_Model, Decoder_Model):
    total_step = len(train_dataloader)

    for i, packs in enumerate(train_dataloader):

        blocks, paths, gts = [], [], []
        for pack in packs:
            image, gt, path = pack["image"], pack["gt"], pack["path"][0]

            if torch.cuda.is_available():
                image = Variable(image.cuda(), requires_grad=False)
            else:
                image = Variable(image, requires_grad=False)

            # 编码
            block = Encoder_Model(image)

            blocks.append(block)
            paths.append(path)
            gts.append(gt)

        # 解码
        out1, out2, out3, out4 = Decoder_Model(blocks)

        # 显示内容
        print('Save_Image: [{:04d}/{:04d}]'.format(i, total_step))
        # 保存图片
        for k in range(len(paths)):
            Save_result(out4[k], paths[k], args.predict_data_path)

    return total_step


if __name__ == '__main__':
    print("start Predicting!!!")

    start_time = time.time()
    step = predict(train_dataloader, Encoder_Model, Decoder_Model)
    end_time = time.time()

    print((end_time - start_time) / step)

    print('-------------Congratulations! Predicting Done!!!-------------')


# 123456
