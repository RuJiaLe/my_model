import torch

from model.utils import self_attention, Video_Encoder_Part
from model.train_model import Video_Encoder_Model, Video_Decoder_Model

x1 = torch.rand(1, 3, 256, 256)
x2 = torch.rand(1, 3, 256, 256)
x3 = torch.rand(1, 3, 256, 256)
x4 = torch.rand(1, 3, 256, 256)

# A = self_attention()
#
# out = A(x1, x2, x3, x4)
# print(out.size())

Encoder_Model = Video_Encoder_Model(output_stride=16, input_channels=12, pretrained=True)
Decoder_Model = Video_Decoder_Model()

x = [x1, x2, x3, x4]
block = Encoder_Model(x)
out = Decoder_Model(block)

print(f'out1_size: {block[0].size()}')
print(f'out2_size: {block[1].size()}')
print(f'out3_size: {block[2].size()}')
print(f'out4_size: {block[3].size()}')

# from model.ConvGRU import ConvGRUCell
# import torch
#
# x1 = torch.rand(1, 3, 256, 256)
#
# model = ConvGRUCell(3, 3)
#
# state = model(x1, x1)
#
# print(state.size())
