import torch
from model.model import Model
import time

model = Model(in_channels=3)


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


start_time = time.time()

x1 = torch.rand(1, 3, 256, 256)
x2 = torch.rand(1, 3, 256, 256)
x3 = torch.rand(1, 3, 256, 256)
x4 = torch.rand(1, 3, 256, 256)
x_s = [x1, x2, x3, x4]

out = model(x_s)

out4, out3, out2, out1, out0 = out

end_time = time.time()

print(f"time_cost: {(end_time - start_time) / 1.0}")

print(f'out4[0]: {out4[0].size(), out4[0].max()}')
print(f'out3[0]: {out3[0].size(), out3[0].max()}')
print(f'out2[0]: {out2[0].size(), out2[0].max()}')
print(f'out1[0]: {out1[0].size(), out1[0].max()}')
print(f'out0[0]: {out0[0].size(), out0[0].max()}')

print(f'Model_parameter: {count_param(model) / 1e6}')
