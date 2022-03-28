import torch
from model.pre_train_model import Video_Encoder_Model, Video_Decoder_Model
import time

Encoder_Model = Video_Encoder_Model(output_stride=16, input_channels=12, pretrained=True)
Decoder_Model = Video_Decoder_Model()


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


start_time = time.time()

# for i in range(10):

x1 = torch.rand(1, 3, 256, 256)
x2 = torch.rand(1, 3, 256, 256)
x3 = torch.rand(1, 3, 256, 256)
x4 = torch.rand(1, 3, 256, 256)
x_s = [x1, x2, x3, x4]
block = Encoder_Model(x_s)
out4, out3, out2, out1, out0 = Decoder_Model(block)

end_time = time.time()

print(f"time_cost: {(end_time - start_time) / 10}")
print(f'out4[0]: {out4[0].size()}')
print(f'out3[0]: {out3[0].size()}')
print(f'out2[0]: {out2[0].size()}')
print(f'out1[0]: {out1[0].size()}')
print(f'out0[0]: {out0[0].size()}')

# print(f'block[0]_size: {block[0].size()}')
# print(f'block[1]_size: {block[1].size()}')
# print(f'block[2]_size: {block[2].size()}')
# print(f'block[3]_size: {block[3].size()}')

print(f'Encoder_Model_parameter: {count_param(Encoder_Model)}')
print(f'Decoder_Model_parameter: {count_param(Decoder_Model)}')
