import os
import torch
from torch.autograd import Variable
import time
import logging

from .Other_material import Save_result


# 开始predict
def predict(dataloader, model, dataset, predict_data_path):
    print('start predict {} !!!'.format(dataset))
    img_num = len(dataloader)
    speed = 0.0

    model.eval()

    for i, packs in enumerate(dataloader):
        i = i + 1
        images, paths, gts = [], [], []
        in_time = time.time()
        for pack in packs:
            image, gt, path = pack["image"], pack["gt"], pack["path"][0]

            if torch.cuda.is_available():
                image = Variable(image.cuda(), requires_grad=False)
            else:
                image = Variable(image, requires_grad=False)

            images.append(image)
            paths.append(path)
            gts.append(gt)

        # 解码
        out4, out3, out2, out1, out0 = model(images)

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


def start_predict(predict_dataloader, model, model_path, dataset, predict_data_path, log_dir):
    logging.basicConfig(filename=log_dir + '/predict_log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    if torch.cuda.is_available():
        model.cuda()

    # 加载模型
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint["state_dict"])

    predict(predict_dataloader, model, dataset, predict_data_path)
