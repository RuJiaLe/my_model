from model.utils import block_aspp_moudle, CS_attention_module
import torch
import time
from model.ConvGRU import ConvGRUCell


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


# y = 0
#
# aspp = block_aspp_moudle(in_dim=2048, out_dim=1024, output_stride=16)
# ATT = CS_attention_module(in_channels=1024)
#
# start = time.time()
#
# for i in range(10):
#     x = torch.rand(1, 1024, 128, 128)
#     # y = aspp(x)
#     y = ATT(x)
#
# end = time.time()
#
# print(y.size())
# print((end - start) / 10)
# print(f'aspp_parameter: {count_param(ATT)}')

out1_1 = torch.rand(1, 1024, 8, 8)
out1_2 = torch.rand(1, 1024, 8, 8)
out1_3 = torch.rand(1, 1024, 8, 8)
out1_4 = torch.rand(1, 1024, 8, 8)

ConvGRU = ConvGRUCell(1024, 1024)

start = time.time()

for i in range(10):
    out0_1 = ConvGRU(out1_1, None)
    out0_2 = ConvGRU(out1_2, out0_1)
    out0_3 = ConvGRU(out1_3, out0_2)
    out0_4 = ConvGRU(out1_4, out0_3)

end = time.time()

print((end - start) / 10.0)
print(f'GRU_parameter: {count_param(ConvGRU)}')
