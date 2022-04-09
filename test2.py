import torch
from utils.Other_material import count_param
import time
from model.triplet_attention import TripletAttention
import math

# from SSA.ssa import shunted_b
#
# x = torch.rand(1, 3, 256, 256)
#
# model = shunted_b()
#
# y = model(x)
#
# print(y.size())
#
# print(f'model_parameters: {count_param(model) / 1e6}')

a = torch.tensor([])

x1 = torch.rand(1, 3, 256, 256)
x2 = torch.rand(1, 3, 256, 256)
x3 = torch.rand(1, 3, 256, 256)
x4 = torch.rand(1, 3, 256, 256)

a += x1
a += x2
a += x3
a += x4

print(len(a))
