import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import resnet50


# CBR2
class CBR2(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(CBR2, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                              padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


# CBR4
class CBR4(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(CBR4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels // 2, kernel_size=(3, 3), stride=(1, 1),
                               padding=1)
        self.bn1 = nn.BatchNorm2d(input_channels // 2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=input_channels // 2, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


# CBS
class CBS(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(CBS, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
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


# attention module模块
class S_attention_module(nn.Module):
    def __init__(self, in_channels):
        super(S_attention_module, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, x):
        spatial_attention = torch.sigmoid(self.conv(x))

        out = x + x * spatial_attention
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

    def __init__(self, output_stride, input_channels=3, pretrained=False):
        super(Video_Encoder_Part, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(3, 3), padding=1,
                               stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.resnet = resnet50(pretrained=pretrained)

        self.block3_aspp = block_aspp_moudle(in_dim=1024, out_dim=512, output_stride=output_stride)
        self.block4_aspp = block_aspp_moudle(in_dim=2048, out_dim=1024, output_stride=output_stride)

        self.conv1x1_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), bias=False)
        self.conv1x1_2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), bias=False)

        self.S_attention_module_1 = S_attention_module(in_channels=128)
        self.S_attention_module_2 = S_attention_module(in_channels=256)
        self.CS_attention_module_3 = CS_attention_module(in_channels=512)
        self.CS_attention_module_4 = CS_attention_module(in_channels=1024)

    def forward(self, x):  # (1, 3, 256, 256)
        block0 = self.conv1(x)
        block0 = self.bn1(block0)
        block0 = self.relu1(block0)
        block0 = self.max_pool1(block0)  # (1, 64, 128, 128)

        block1 = self.resnet.layer1(block0)  # (1, 256, 128, 128)
        block2 = self.resnet.layer2(block1)  # (1, 512, 64, 64)
        block3 = self.resnet.layer3(block2)  # (1, 1024, 32, 32)
        block4 = self.resnet.layer4(block3)  # (1, 2048, 16, 16)

        block1_result = self.conv1x1_1(block1)
        block2_result = self.conv1x1_2(block2)
        block3_result = self.block3_aspp(block3)
        block4_result = self.block4_aspp(block4)

        block1_result = self.S_attention_module_1(block1_result)  # (1, 128, 128, 128)
        block2_result = self.S_attention_module_2(block2_result)  # (1, 256, 64, 64)
        bloch3_result = self.CS_attention_module_3(block3_result)  # (1, 512, 32, 32)
        bloch4_result = self.CS_attention_module_4(block4_result)  # (1, 1024, 16, 16)

        return [block1_result, block2_result, bloch3_result, bloch4_result]


# multi_headed_self_attention_module
class self_attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(self_attention, self).__init__()
        self.CBR4_1 = CBR2(in_channels, out_channels)
        self.CBR4_2 = CBR2(in_channels, out_channels)
        self.CBR4_3 = CBR2(in_channels, out_channels)
        self.CBR4_4 = CBR2(in_channels, out_channels)

        self.conv1 = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=(1, 1), bias=False)

        self.Linear = nn.Linear(in_channels, in_channels // 2)

        self.Up_sample_2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x4_1, x4_2, x4_3, x4_4):  # (1, 1024, 16, 16)

        x4_1 = self.CBR4_1(x4_1)  # (1, 512, 16, 16)
        x4_2 = self.CBR4_2(x4_2)
        x4_3 = self.CBR4_3(x4_3)
        x4_4 = self.CBR4_4(x4_4)

        attention_feature = torch.cat((x4_1, x4_2, x4_3, x4_4), dim=1)
        attention_feature = self.conv1(attention_feature)

        N, C, H, W = attention_feature.shape
        attention_feature_max = F.adaptive_max_pool2d(attention_feature, (1, 1)).view(N, C) / math.sqrt(C)

        attention = F.softmax(self.Linear(attention_feature_max), dim=1)

        w1 = w2 = w3 = w4 = attention
        w = torch.cat((w1, w2, w3, w4), dim=0) / math.sqrt(C)
        w = F.softmax(w, dim=1)
        w = torch.mean(w, dim=1)
        w1, w2, w3, w4 = w[0], w[1], w[2], w[3]

        attention1 = attention * w1
        attention2 = attention * w2
        attention3 = attention * w3
        attention4 = attention * w4

        out1 = x4_1 + x4_1 * attention1.unsqueeze(2).unsqueeze(3)
        out2 = x4_2 + x4_2 * attention2.unsqueeze(2).unsqueeze(3)
        out3 = x4_3 + x4_3 * attention3.unsqueeze(2).unsqueeze(3)
        out4 = x4_4 + x4_4 * attention4.unsqueeze(2).unsqueeze(3)

        out1 = self.Up_sample_2(out1)
        out2 = self.Up_sample_2(out2)
        out3 = self.Up_sample_2(out3)
        out4 = self.Up_sample_2(out4)

        return out1, out2, out3, out4


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
