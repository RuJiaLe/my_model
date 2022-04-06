import argparse
from torch.utils.data import DataLoader
from dataset.dataload import ImageDataset
from dataset.dataload import VideoDataset
from dataset.dataload import EvalDataset
from dataset.transforms import get_train_transforms, get_transforms, get_Eval_transforms

from model.video_train_model import Video_Encoder_Model, Video_Decoder_Model
from model.image_train_model import Image_Encoder_Model, Image_Decoder_Model
from utils.Train_material import start_image_train, start_video_train
from utils.Predict_material import start_predict
from utils.Evaluation_material import start_Eval

parser = argparse.ArgumentParser()

# parameters
parser.add_argument('--epoch', type=int, default=60, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
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
parser.add_argument('--image_encoder_model', type=str, default="./save_models/image_train_model/best_image_encoder_model.pth.tar", help='image_encoder_model')
parser.add_argument('--image_decoder_model', type=str, default="./save_models/image_train_model/best_image_decoder_model.pth.tar", help='image_decoder_model')

parser.add_argument('--video_encoder_model', type=str, default="./save_models/video_train_model/best_video_encoder_model.pth.tar", help='video_encoder_model')
parser.add_argument('--video_decoder_model', type=str, default="./save_models/video_train_model/best_video_decoder_model.pth.tar", help='video_decoder_model')

args = parser.parse_args()


# image_train
def image_train():
    print("start image training!!!")
    image_encoder_model = Image_Encoder_Model(output_stride=16, input_channels=3, pretrained=True)
    image_decoder_model = Image_Decoder_Model()

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
    print(f'load image val data done, total val number is {len(image_val_dataloader) * 4}')

    start_image_train(image_train_dataloader, image_val_dataloader, image_encoder_model, image_decoder_model, args.epoch, args.lr, args.decay_rate,
                      args.decay_epoch, args.image_encoder_model, args.image_decoder_model, args.log_dir, args.start_epoch)

    print('-------------Congratulations! Image Training Done!!!-------------')


# video_train
def video_train():
    print("start video training!!!")
    video_encoder_model = Video_Encoder_Model(output_stride=16, input_channels=3, pretrained=True)
    video_decoder_model = Video_Decoder_Model()

    # 数据加载
    # train data load
    video_train_transforms = get_train_transforms(input_size=(args.size, args.size))
    video_train_dataset = VideoDataset(root_dir=args.video_train_path, training_set_list=args.video_train_dataset, training=True,
                                       transforms=video_train_transforms)
    video_train_dataloader = DataLoader(dataset=video_train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    # val data load
    video_val_transforms = get_transforms(input_size=(args.size, args.size))
    video_val_dataset = VideoDataset(root_dir=args.video_val_path, training_set_list=args.video_val_dataset, training=True,
                                     transforms=video_val_transforms)
    video_val_dataloader = DataLoader(dataset=video_val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=True)

    print(f'load video train data done, total train number is {len(video_train_dataloader)}')
    print(f'load video val data done, total val number is {len(video_val_dataloader)}')

    start_video_train(video_train_dataloader, video_val_dataloader, video_encoder_model, video_decoder_model, args.epoch, args.lr, args.decay_rate,
                      args.decay_epoch, args.video_encoder_model, args.video_decoder_model, args.log_dir, args.start_epoch)

    print('-------------Congratulations! Image Training Done!!!-------------')


# predict
def predict():
    print("start video predict!!!")
    video_encoder_model = Video_Encoder_Model(output_stride=16, input_channels=3, pretrained=True)
    video_decoder_model = Video_Decoder_Model()

    # 数据加载
    predict_transforms = get_transforms(input_size=(args.size, args.size))
    predict_dataset = VideoDataset(root_dir=args.predict_data_path, training_set_list=[args.predict_dataset], training=True,
                                   transforms=predict_transforms)
    predict_dataloader = DataLoader(dataset=predict_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=True)

    print(f'load predict data done, total train number is {len(predict_dataloader) * 4}')

    start_predict(predict_dataloader, video_encoder_model, video_decoder_model, args.video_encoder_model, args.video_decoder_model,
                  args.predict_dataset, args.predict_data_path, args.log_dir)

    print('-------------Congratulations! video predict Done!!!-------------')


# Eval
def Evaluation():
    print("start Evaluation!!!")

    Eval_transforms = get_Eval_transforms(input_size=(args.size, args.size))
    Eval_dataset = EvalDataset(root_dir=args.predict_data_path, training_set_list=[args.predict_dataset], training=False,
                               transforms=Eval_transforms)
    Eval_dataloader = DataLoader(dataset=Eval_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=True)

    print(f'load evaluation data done, total train number is {len(Eval_dataloader) * 4}')

    start_Eval(args.predict_dataset, Eval_dataloader, args.log_dir)

    print('-------------Congratulations! Evaluation Done!!!-------------')


def main():
    model = input('please select model from [image_train, video_train, predict, Evaluation]:')

    if model == 'image_train':
        image_train()

    elif model == 'video_train':
        video_train()

    elif model == 'predict':
        predict()

    elif model == 'Evaluation':
        Evaluation()

    else:
        print('select error')


if __name__ == '__main__':
    main()
