import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import resnet50


# CBR2
class CR2(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(CR2, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                              padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)

        return x


# CBR4
class CR4(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(CR4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels // 2, kernel_size=(3, 3), stride=(1, 1),
                               padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=input_channels // 2, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                               padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        return x


# CBS
class CS(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(CS, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=(1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)

        return x


# ASPP模块
class block_aspp_moudle(nn.Module):
    def __init__(self, in_dim, out_dim, output_stride=16, rates=[6, 12, 18]):
        super(block_aspp_moudle, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []

        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=(1, 1), bias=False),
                          nn.ReLU(inplace=True))
        )

        # other rates
        for r in rates:
            self.features.append(
                nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=(3, 3),
                                        dilation=(r, r), padding=r, bias=False),
                              nn.ReLU(inplace=True)
                              )
            )

        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=(1, 1), bias=False),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Conv2d(in_channels=out_dim * 5, out_channels=out_dim, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:], mode='bilinear', align_corners=True)

        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)

        out = self.fuse(out)

        return out


class CS_attention_module(nn.Module):
    def __init__(self, in_channels):
        super(CS_attention_module, self).__init__()
        self.Linear = nn.Linear(in_channels, in_channels)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, x):
        N, C, H, W = x.shape
        attention_feature_max = F.adaptive_max_pool2d(x, (1, 1)).view(N, C) / math.sqrt(C)
        channel_attention = F.softmax(self.Linear(attention_feature_max), dim=1).unsqueeze(2).unsqueeze(3)
        spatial_attention = torch.sigmoid(self.conv(x))

        out = x + x * channel_attention * spatial_attention

        return out


class Video_Encoder_Part(nn.Module):

    def __init__(self, output_stride, input_channels=12, pretrained=False):
        super(Video_Encoder_Part, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(7, 7), padding=3,
                               stride=(2, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.resnet = resnet50(pretrained=pretrained)

        self.block1_aspp = block_aspp_moudle(in_dim=256, out_dim=256, output_stride=output_stride)
        self.block2_aspp = block_aspp_moudle(in_dim=512, out_dim=512, output_stride=output_stride)
        self.block3_aspp = block_aspp_moudle(in_dim=1024, out_dim=1024, output_stride=output_stride)
        self.block4_aspp = block_aspp_moudle(in_dim=2048, out_dim=2048, output_stride=output_stride)

        self.CS_attention_module_1 = CS_attention_module(in_channels=256)
        self.CS_attention_module_2 = CS_attention_module(in_channels=512)
        self.CS_attention_module_3 = CS_attention_module(in_channels=1024)
        self.CS_attention_module_4 = CS_attention_module(in_channels=2048)

    def forward(self, x):  # (1, 12, 256, 256)
        block0 = self.conv1(x)  # (1, 64, 128, 128)
        block0 = self.bn1(block0)
        block0 = self.relu1(block0)
        block0 = self.max_pool1(block0)  # (1, 64, 64, 64)

        block1 = self.resnet.layer1(block0)  # (1, 256, 64, 64)
        block2 = self.resnet.layer2(block1)  # (1, 512, 32, 32)
        block3 = self.resnet.layer3(block2)  # (1, 1024, 16, 16)
        block4 = self.resnet.layer4(block3)  # (1, 2048, 8, 8)

        # print(f'block1: {block1.size()}')
        # print(f'block2: {block2.size()}')
        # print(f'block3: {block3.size()}')
        # print(f'block4: {block4.size()}')

        block1_result = self.block1_aspp(block1)
        block2_result = self.block2_aspp(block2)
        block3_result = self.block3_aspp(block3)
        block4_result = self.block4_aspp(block4)

        bloch1_result = self.CS_attention_module_1(block1_result)  # (1, 512, 32, 32)
        bloch2_result = self.CS_attention_module_2(block2_result)  # (1, 1024, 16, 16)
        bloch3_result = self.CS_attention_module_3(block3_result)  # (1, 512, 32, 32)
        bloch4_result = self.CS_attention_module_4(block4_result)  # (1, 1024, 16, 16)

        return [block1_result, block2_result, bloch3_result, bloch4_result]


# Video_Decoder_Part
class Video_Decoder_Part(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Video_Decoder_Part, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))

        self.Up_sample_2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)

        x = self.conv1x1(x)
        out = out * x

        out = self.Up_sample_2(out)

        return out


# multi_headed_self_attention_module
class self_attention(nn.Module):
    def __init__(self):
        super(self_attention, self).__init__()

    def forward(self, frames):  # (1, 3, 256, 256)
        x1, x2, x3, x4 = frames[0], frames[1], frames[2], frames[3]
        V = torch.cat((x1, x2, x3, x4), dim=1)
        Q1 = torch.cat([x1] * 4, dim=1)
        Q2 = torch.cat([x2] * 4, dim=1)
        Q3 = torch.cat([x3] * 4, dim=1)
        Q4 = torch.cat([x4] * 4, dim=1)

        alpha1 = Q1 * V
        alpha2 = Q2 * V
        alpha3 = Q3 * V
        alpha4 = Q4 * V

        out1 = F.softmax(alpha1, dim=1) * V
        out2 = F.softmax(alpha2, dim=1) * V
        out3 = F.softmax(alpha3, dim=1) * V
        out4 = F.softmax(alpha4, dim=1) * V

        out1 = torch.chunk(out1, 4, dim=1)
        out2 = torch.chunk(out2, 4, dim=1)
        out3 = torch.chunk(out3, 4, dim=1)
        out4 = torch.chunk(out4, 4, dim=1)

        out1 = out1[0] + out1[1] + out1[2] + out1[3]
        out2 = out2[0] + out2[1] + out2[2] + out2[3]
        out3 = out3[0] + out3[1] + out3[2] + out3[3]
        out4 = out4[0] + out4[1] + out4[2] + out4[3]

        out = torch.cat((out1, out2, out3, out4), dim=1)

        return out


# pre_train
class pre_train_decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(pre_train_decoder, self).__init__()
        self.CBR4_1 = CBR2(in_channels, out_channels)
        self.CBR4_2 = CBR2(in_channels, out_channels)
        self.CBR4_3 = CBR2(in_channels, out_channels)
        self.CBR4_4 = CBR2(in_channels, out_channels)

        self.Up_sample_2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x4_1, x4_2, x4_3, x4_4):  # (1, 1024, 16, 16)

        x4_1 = self.CBR4_1(x4_1)  # (1, 512, 16, 16)
        x4_2 = self.CBR4_2(x4_2)
        x4_3 = self.CBR4_3(x4_3)
        x4_4 = self.CBR4_4(x4_4)

        out1 = self.Up_sample_2(x4_1)
        out2 = self.Up_sample_2(x4_2)
        out3 = self.Up_sample_2(x4_3)
        out4 = self.Up_sample_2(x4_4)

        return out1, out2, out3, out4
