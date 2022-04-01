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
import logging

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--size', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--save_model', type=str, default="./save_models/pretrain_model", help='save_model')
parser.add_argument('--predict_data_path', type=str, default="./data/test_predict_data", help='predict_data_path')
parser.add_argument('--dataset', type=list, default=["Test"], help='dataset')
parser.add_argument('--log_dir', type=str, default="./Log_file", help="log_dir file")

args = parser.parse_args()

logging.basicConfig(filename=args.log_dir + '/pre_predict_log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

# 模型加载
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

Encoder_Model = Video_Encoder_Model(output_stride=16, input_channels=3, pretrained=False)
Decoder_Model = Video_Decoder_Model()

Encoder_path = args.save_model + "/best_pre_encoder_model.pth"
Decoder_path = args.save_model + "/best_pre_decoder_model.pth"
if os.path.exists(Encoder_path):
    print('Loading state dict from: {}'.format(Encoder_path))
    Encoder_Model.load_state_dict(torch.load(Encoder_path, map_location=torch.device(device)))
    print('Loading state dict from: {}'.format(Decoder_path))
    Decoder_Model.load_state_dict(torch.load(Decoder_path, map_location=torch.device(device)))

if torch.cuda.is_available():
    Encoder_Model.cuda()
    Decoder_Model.cuda()

# 开始predict
Encoder_Model.eval()
Decoder_Model.eval()


def predict(dataloader, encoder_model, decoder_model, dataset):
    print('start predict {} !!!'.format(dataset))
    img_num = len(dataloader)
    speed = 0.0

    for i, packs in enumerate(dataloader):
        i = i + 1
        images, paths, gts, blocks = [], [], [], []
        in_time = time.time()
        for pack in packs:
            image, gt, path = pack["image"], pack["gt"], pack["path"][0]

            if torch.cuda.is_available():
                image = Variable(image.cuda(), requires_grad=False)
            else:
                image = Variable(image, requires_grad=False)

            block = encoder_model(image)

            blocks.append(block)
            images.append(image)
            paths.append(path)
            gts.append(gt)

        # 编码 解码
        out4, out3, out2, out1, out0 = decoder_model(blocks)

        # 显示内容
        out_time = time.time()
        speed += (out_time - in_time)
        print('dataset:{}, done: {:0.2f}%, Save_Image: {}/{}, speed:{:0.4f}'.format(dataset, (i / img_num) * 100, i, img_num, out_time - in_time))
        logging.info('dataset:{}, done: {:0.2f}%, Save_Image: {}/{}, speed:{:0.4f}'.format(dataset, (i / img_num) * 100, i, img_num, out_time - in_time))

        # 保存图片
        for k in range(len(paths)):
            Save_result(out0[k], paths[k], args.predict_data_path)

    print('dataset: {}, img_num:{}, avg_speed: {:0.4f}'.format(dataset, img_num * 4, speed / (img_num * 4)))
    logging.info('dataset: {}, total_num:{}, avg_speed: {:0.4f}'.format(dataset, img_num * 4, speed / (img_num * 4)))


if __name__ == '__main__':
    print("start Predicting!!!")

    # 数据加载
    for data in args.dataset:
        if data:
            transforms = get_transforms(input_size=(args.size, args.size))
            pre_predict_dataset = VideoDataset(root_dir=args.predict_data_path, training_set_list=[data], training=True,
                                               transforms=transforms)
            pre_predict_dataloader = DataLoader(dataset=pre_predict_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=True)

            predict(pre_predict_dataloader, Encoder_Model, Decoder_Model, data)

    print('-------------Congratulations! Predicting Done!!!-------------')
