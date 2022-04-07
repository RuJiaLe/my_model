import torch
from utils.Other_material import count_param
import time
from model.triplet_attention import TripletAttention

x = torch.rand(5, 3, 256, 256)

for i in range(5):
    y = x[i, ...].unsqueeze(0)

    print(y.size())

# print(f'block1_size: {block[0].size()}')
# print(f'block2_size: {block[1].size()}')
# print(f'block3_size: {block[2].size()}')
# print(f'block4_size: {block[3].size()}')
