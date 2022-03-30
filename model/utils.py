import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Res_Block
class Res_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding=8, dilation=(8, 8))
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=4, dilation=(4, 4))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=2, dilation=(2, 2))
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        x = self.conv1x1(x)

        out = out + x
        out = self.relu2(out)

        out = self.max_pool(out)

        return out


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


# Video_Decoder_Part
class Video_Decoder_Part(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Video_Decoder_Part, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1))
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1))
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

# # multi_headed_self_attention_module
# class self_attention(nn.Module):
#     def __init__(self):
#         super(self_attention, self).__init__()
#
#     def forward(self, frames):  # (1, 3, 256, 256)
#         x1, x2, x3, x4 = frames[0], frames[1], frames[2], frames[3]
#         V = torch.cat((x1, x2, x3, x4), dim=1)
#         Q1 = torch.cat([x1] * 4, dim=1)
#         Q2 = torch.cat([x2] * 4, dim=1)
#         Q3 = torch.cat([x3] * 4, dim=1)
#         Q4 = torch.cat([x4] * 4, dim=1)
#
#         alpha1 = Q1 * V
#         alpha2 = Q2 * V
#         alpha3 = Q3 * V
#         alpha4 = Q4 * V
#
#         out1 = F.softmax(alpha1, dim=1) * V
#         out2 = F.softmax(alpha2, dim=1) * V
#         out3 = F.softmax(alpha3, dim=1) * V
#         out4 = F.softmax(alpha4, dim=1) * V
#
#         out1 = torch.chunk(out1, 4, dim=1)
#         out2 = torch.chunk(out2, 4, dim=1)
#         out3 = torch.chunk(out3, 4, dim=1)
#         out4 = torch.chunk(out4, 4, dim=1)
#
#         out1 = out1[0] + out1[1] + out1[2] + out1[3]
#         out2 = out2[0] + out2[1] + out2[2] + out2[3]
#         out3 = out3[0] + out3[1] + out3[2] + out3[3]
#         out4 = out4[0] + out4[1] + out4[2] + out4[3]
#
#         out = torch.cat((out1, out2, out3, out4), dim=1)
#
#         return out
