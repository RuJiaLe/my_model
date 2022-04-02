import torch
import time
from torch.autograd import Variable
from utils.Other_material import Eval_mae, Eval_F_measure, Eval_E_measure, Eval_S_measure
import logging


# Eval
def Eval(dataset, dataloader):
    total_num = len(dataloader) * 4
    MAES, E_measures, S_measures = 0.0, 0.0, 0.0
    img_num = 0
    avg_p, avg_r = 0.0, 0.0

    start_time = time.time()
    for j, packs in enumerate(dataloader):
        for pack in packs:
            img_num = img_num + 1
            predict, gt = pack['predict'], pack['gt']
            if torch.cuda.is_available():
                predict, gt = Variable(predict.cuda(), requires_grad=False), Variable(gt.cuda(), requires_grad=False)
            else:
                predict, gt = Variable(predict, requires_grad=False), Variable(gt, requires_grad=False)

            mae = Eval_mae(predict, gt)
            MAES += mae.data

            prec, recall = Eval_F_measure(predict, gt)
            avg_p += prec.data
            avg_r += recall.data

            E_measure = Eval_E_measure(predict, gt)
            E_measures += E_measure.data
            S_measure = Eval_S_measure(predict, gt)
            S_measures += S_measure.data

        if img_num % 1 == 0:
            print('dataset: {}, done: {:0.2f}%, img: {}/{}, mae: {:0.4f}, E_measure: {:0.4f}, S_measure: {:0.4f}'.
                  format(dataset, (img_num / total_num) * 100, img_num, total_num, MAES / img_num, E_measures / img_num, S_measures / img_num))

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
    speed = end_time - start_time

    print('dataset: {}, total_img: {}, avg_mae: {:0.4f}, max_F_measure: {:0.4f}, avg_E_measure: {:0.4f}, avg_S_measure: {:0.4f}, speed: {:0.4f}'.
          format(dataset, img_num, avg_mae, max_F_measure, avg_E_measure, avg_S_measure, speed))

    logging.info('{}'.format('*' * 100))
    logging.info('dataset: {}, total_img: {} avg_mae: {:0.4f}, max_F_measure: {:0.4f}, avg_E_measure: {:0.4f}, avg_S_measure: {:0.4f}, speed: {:0.4f}'.
                 format(dataset, img_num, avg_mae, max_F_measure, avg_E_measure, avg_S_measure, speed))
    logging.info('{}'.format('*' * 100))


def start_Eval(data, Eval_dataloader, log_dir):
    logging.basicConfig(filename=log_dir + '/val_log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    Eval(data, Eval_dataloader)
