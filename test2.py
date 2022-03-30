import torch
from model.Resnet import resnet50, resnet18
from model.train_model import Video_Encoder_Model
from model.utils import block_aspp_moudle
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet34
import time
from model.resnet_dilation import resnet34


#
#
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


# resnet34 = resnet34(pretrained=True)
model = Video_Encoder_Model(output_stride=16, input_channels=3, pretrained=True)

start = time.time()

for i in range(10):
    x = torch.rand(1, 3, 256, 256)
    block1, block2, block3, block4, block5 = model(x)

end = time.time()

print(f'block1: {block1.size()}')
print(f'block2: {block2.size()}')
print(f'block3: {block3.size()}')
print(f'block4: {block4.size()}')
print(f'block5: {block5.size()}')

# print(f'resnet18_parameter: {count_param(resnet18) / 1e6}')
# print(f'resnet34_parameter: {count_param(resnet34) / 1e6}')
print(f'model_parameter: {count_param(model) / 1e6}')

print((end - start) / 10.0)
#
# x = torch.rand(1, 64, 128, 128)
#
# Res = resnet34(pretrained=True)
#
# block1 = Res.layer1(x)
# block2 = Res.layer2(block1)
# block3 = Res.layer3(block2)
# block4 = Res.layer4(block3)
#
# print(f'block1: {block1.size()}')
# print(f'block2: {block2.size()}')
# print(f'block3: {block3.size()}')
# print(f'block4: {block4.size()}')
#
# print(f'Res_parameter: {count_param(Res) / 1e6}')
