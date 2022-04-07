import torch
from model.video_train_model import Video_Encoder_Model, Video_Decoder_Model
from model.model import Model
import time

Encoder_Model = Video_Encoder_Model(output_stride=16, input_channels=3, pretrained=True)
Decoder_Model = Video_Decoder_Model()

model = Model(in_channels=3)


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
# blocks = []
# for x in x_s:
#     block = Encoder_Model(x)
#     blocks.append(block)
# out4, out3, out2, out1, out0 = Decoder_Model(blocks)

out4, out3, out2, out1, out0 = model(x_s)

end_time = time.time()

print(f"time_cost: {(end_time - start_time) / 1.0}")

print(f'out3[0]: {out4[0].size()}')
print(f'out2[0]: {out3[0].size()}')
print(f'out1[0]: {out2[0].size()}')
print(f'out0[0]: {out1[0].size()}')

# print(f'block[0]_size: {block[0].size()}')
# print(f'block[1]_size: {block[1].size()}')
# print(f'block[2]_size: {block[2].size()}')
# print(f'block[3]_size: {block[3].size()}')

# print(f'Encoder_Model_parameter: {count_param(Encoder_Model) / 1e6}')  # 25.560388  17.833s
# print(f'Decoder_Model_parameter: {count_param(Decoder_Model) / 1e6}')  # 69.930239

print(f'Model_parameter: {count_param(model) / 1e6}')
