import os
import torch
from torch.autograd import Variable
import argparse
from dataset.dataload import ValDataset
from dataset.transforms import get_val_transforms
from torch.utils.data import DataLoader
from utils import Eval_mae, Eval_F_measure, Eval_E_measure, Eval_S_measure
import logging
from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--size', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--predict_data_path', type=str, default="./predict_data", help='predict_data_path')
parser.add_argument('--datasets', type=list, default=["DAVIS", "DAVSOD", "DAVSOD_D", "DAVSOD_E", "DAVSOD_N", "SegTrackV2", "UVSD"], help='datasets')
parser.add_argument('--log_dir', type=str, default="./Log_file", help="log_dir file")

args = parser.parse_args()

for i in range(len(args.datasets)):
    logging.basicConfig(filename=args.log_dir + '/val_log_34th.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # 数据加载
    transforms = get_val_transforms(input_size=(args.size, args.size))
    dataset = ValDataset(root_dir=args.predict_data_path, training_set_list=[args.datasets[i]], training=False,
                         transforms=transforms)
    Val_dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=True)

    MAES, F_measures, E_measures, S_measures = [], [], [], []

    for j, packs in enumerate(Val_dataloader):
        for pack in packs:
            predict, gt = pack['predict'], pack['gt']
            if torch.cuda.is_available():
                predict, gt = Variable(predict.cuda(), requires_grad=False), Variable(gt.cuda(), requires_grad=False)
            else:
                predict, gt = Variable(predict, requires_grad=False), Variable(gt, requires_grad=False)

            mae = Eval_mae(predict, gt)
            F_measure = Eval_F_measure(predict, gt)
            E_measure = Eval_E_measure(predict, gt)
            S_measure = Eval_S_measure(predict, gt)

            MAES.append(mae)
            F_measures.append(F_measure)
            E_measures.append(E_measure)
            S_measures.append(S_measure)

            print('dataset: {}, mae: {:0.4f}, F_measure: {:0.4f}, E_measure: {:0.4f}, S_measure: {:0.4f}'.format(args.datasets[i], mae, F_measure, E_measure, S_measure))
            logging.info('dataset: {}, mae: {:0.4f}, F_measure: {:0.4f}, E_measure: {:0.4f}, S_measure: {:0.4f}'.format(args.datasets[i], mae, F_measure, E_measure, S_measure))

    avg_mae = torch.mean(torch.tensor(MAES))
    avg_F_measure = torch.mean(torch.tensor(F_measures))
    avg_E_measure = torch.mean(torch.tensor(E_measures))
    avg_S_measure = torch.mean(torch.tensor(S_measures))

    max_mae = torch.max(torch.tensor(MAES))
    max_F_measure = torch.max(torch.tensor(F_measures))
    max_E_measure = torch.max(torch.tensor(E_measures))
    max_S_measure = torch.max(torch.tensor(S_measures))

    min_mae = torch.min(torch.tensor(MAES))
    min_F_measure = torch.min(torch.tensor(F_measures))
    min_E_measure = torch.min(torch.tensor(E_measures))
    min_S_measure = torch.min(torch.tensor(S_measures))

    print('dataset: {}, avg_mae: {:0.4f}, avg_F_measure: {:0.4f}, avg_E_measure: {:0.4f}, avg_S_measure: {:0.4f}'.
          format(args.datasets[i], avg_mae, avg_F_measure, avg_E_measure, avg_S_measure))
    print('dataset: {}, max_mae: {:0.4f}, max_F_measure: {:0.4f}, max_E_measure: {:0.4f}, max_S_measure: {:0.4f}'.
          format(args.datasets[i], max_mae, max_F_measure, max_E_measure, max_S_measure))
    print('dataset: {}, min_mae: {:0.4f}, min_F_measure: {:0.4f}, min_E_measure: {:0.4f}, min_S_measure: {:0.4f}'.
          format(args.datasets[i], min_mae, min_F_measure, min_E_measure, min_S_measure))

    logging.info('{}'.format('*' * 50))
    logging.info('dataset: {}, avg_mae: {:0.4f}, avg_F_measure: {:0.4f}, avg_E_measure: {:0.4f}, avg_S_measure: {:0.4f}'.
                 format(args.datasets[i], avg_mae, avg_F_measure, avg_E_measure, avg_S_measure))
    logging.info('dataset: {}, max_mae: {:0.4f}, max_F_measure: {:0.4f}, max_E_measure: {:0.4f}, max_S_measure: {:0.4f}'.
                 format(args.datasets[i], max_mae, max_F_measure, max_E_measure, max_S_measure))
    logging.info('dataset: {}, min_mae: {:0.4f}, min_F_measure: {:0.4f}, min_E_measure: {:0.4f}, min_S_measure: {:0.4f}'.
                 format(args.datasets[i], min_mae, min_F_measure, min_E_measure, min_S_measure))
    logging.info('{}'.format('*' * 50))
