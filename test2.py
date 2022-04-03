import torch
from model.video_train_model import Video_Encoder_Model
from utils.Other_material import count_param
import time
from model.triplet_attention import TripletAttention

encoder_model = Video_Encoder_Model(output_stride=16, input_channels=3, pretrained=True)

x = torch.rand(1, 3, 256, 256)

start = time.time()

block = encoder_model(x)

end = time.time()

print(end - start)

print(f'block1_size: {block[0].size()}')
print(f'block2_size: {block[1].size()}')
print(f'block3_size: {block[2].size()}')
print(f'block4_size: {block[3].size()}')
print(f'model_param: {count_param(encoder_model) / 1e6}')
