import os
import torch
from torch.autograd import Variable
import time
import logging

from .Other_material import Save_result


# 开始predict
def predict(dataloader, encoder_model, decoder_model, dataset, predict_data_path):
    print('start predict {} !!!'.format(dataset))
    img_num = len(dataloader)
    speed = 0.0

    encoder_model.eval()
    decoder_model.eval()

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

            images.append(image)
            paths.append(path)
            gts.append(gt)
            blocks.append(block)

        # 解码
        out4, out3, out2, out1, out0 = decoder_model(blocks)

        # 显示内容
        out_time = time.time()
        speed += (out_time - in_time)
        print('dataset:{}, done: {:0.2f}%, Save_Image: {}/{}, speed:{:0.4f}'.format(dataset, (i / img_num) * 100, i, img_num, out_time - in_time))

        # 保存图片
        for k in range(len(paths)):
            Save_result(out0[k], paths[k], predict_data_path)

    print('dataset: {}, img_num:{}, avg_speed: {:0.4f}'.format(dataset, img_num * 4, speed / (img_num * 4)))
    logging.info('dataset: {}, total_num:{}, avg_speed: {:0.4f}'.format(dataset, img_num * 4, speed / (img_num * 4)))


# 模型加载
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def start_predict(predict_dataloader, encoder_model, decoder_model, encoder_model_path, decoder_model_path, dataset, predict_data_path, log_dir):
    logging.basicConfig(filename=log_dir + '/predict_log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    if os.path.exists(encoder_model_path):
        print('Loading state dict from: {}'.format(encoder_model_path))
        encoder_model.load_state_dict(torch.load(encoder_model_path, map_location=torch.device(device)))
    if os.path.exists(decoder_model_path):
        print('Loading state dict from: {}'.format(decoder_model_path))
        decoder_model.load_state_dict(torch.load(decoder_model_path, map_location=torch.device(device)))

    if torch.cuda.is_available():
        encoder_model.cuda()
        decoder_model.cuda()

    predict(predict_dataloader, encoder_model, decoder_model, dataset, predict_data_path)
