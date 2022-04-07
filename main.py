import argparse
from torch.utils.data import DataLoader
from dataset.dataload import ImageDataset
from dataset.dataload import VideoDataset
from dataset.dataload import EvalDataset
from dataset.transforms import get_train_transforms, get_transforms, get_Eval_transforms

from model.model import Model
from utils.Train_material import start_train
from utils.Predict_material import start_predict
from utils.Evaluation_material import start_Eval

parser = argparse.ArgumentParser()

# parameters
parser.add_argument('--epoch', type=int, default=60, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=2, help='training batch size')
parser.add_argument('--size', type=int, default=256, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.5, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=8, help='every n epochs decay learning rate')
parser.add_argument('--log_dir', type=str, default="./Log_file", help="log_dir file")
parser.add_argument('--start_epoch', type=int, default=1, help='start_epoch')

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
parser.add_argument('--save_model_path', type=str, default="./save_models/best_model.pth.tar", help='save_model_path')

args = parser.parse_args()


# image_train
def train():
    model = Model(in_channels=3)

    flag = input('select train model (image or video):')

    train_dataloader = None
    val_dataloader = None

    if flag == 'image':
        print("start image training!!!")
        # 数据加载
        # train data load
        train_transforms = get_train_transforms(input_size=(args.size, args.size))
        train_dataset = ImageDataset(root_dir=args.image_train_path, training_set_list=args.image_train_dataset,
                                     image_transform=train_transforms)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

        # val data load
        val_transforms = get_transforms(input_size=(args.size, args.size))
        val_dataset = ImageDataset(root_dir=args.image_val_path, training_set_list=args.image_val_dataset,
                                   image_transform=val_transforms)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=True)

    elif flag == 'video':
        print("start video training!!!")
        train_transforms = get_train_transforms(input_size=(args.size, args.size))
        train_dataset = VideoDataset(root_dir=args.video_train_path, training_set_list=args.video_train_dataset, training=True,
                                     transforms=train_transforms)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

        # val data load
        val_transforms = get_transforms(input_size=(args.size, args.size))
        val_dataset = VideoDataset(root_dir=args.video_val_path, training_set_list=args.video_val_dataset, training=True,
                                   transforms=val_transforms)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=True)

    print(f'load train data done, total train number is {len(train_dataloader) * 4 * args.batch_size}')
    print(f'load val data done, total val number is {len(val_dataloader) * 4 * args.batch_size}')

    start_train(train_dataloader, val_dataloader, model, args.epoch, args.lr, args.decay_rate,
                args.decay_epoch, args.save_model_path, args.log_dir, args.start_epoch, args.batch_size)

    print('-------------Congratulations! Training Done!!!-------------')


# predict
def predict():
    print("start video predict!!!")
    model = Model(in_channels=3)

    # 数据加载
    predict_transforms = get_transforms(input_size=(args.size, args.size))
    predict_dataset = VideoDataset(root_dir=args.predict_data_path, training_set_list=[args.predict_dataset], training=True,
                                   transforms=predict_transforms)
    predict_dataloader = DataLoader(dataset=predict_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=True)

    print(f'load predict data done, total train number is {len(predict_dataloader) * 4 * args.batch_size}')

    start_predict(predict_dataloader, model, args.save_model_path, args.predict_dataset, args.predict_data_path, args.log_dir, args.batch_size)

    print('-------------Congratulations! video predict Done!!!-------------')


# Eval
def Evaluation():
    print("start Evaluation!!!")

    Eval_transforms = get_Eval_transforms(input_size=(args.size, args.size))
    Eval_dataset = EvalDataset(root_dir=args.predict_data_path, training_set_list=[args.predict_dataset], training=False,
                               transforms=Eval_transforms)
    Eval_dataloader = DataLoader(dataset=Eval_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=True)

    print(f'load evaluation data done, total train number is {len(Eval_dataloader) * 4 * args.batch_size}')

    start_Eval(args.predict_dataset, Eval_dataloader, args.log_dir, args.batch_size)

    print('-------------Congratulations! Evaluation Done!!!-------------')


def main():
    model = input('please select model from [train, predict, Evaluation]:')

    if model == 'train':
        train()

    elif model == 'predict':
        predict()

    elif model == 'Evaluation':
        Evaluation()

    else:
        print('select error')


if __name__ == '__main__':
    main()
