import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .triplet_attention import TripletAttention


# Res_Block
class En_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(En_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.attention1 = TripletAttention()

        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)

        x = self.conv1x1(x)

        out = out + x
        out = self.relu1(out)

        out = self.max_pool(out)

        out = self.attention1(out)

        return out


# De_block
class De_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(De_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.attention1 = TripletAttention()

        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))

        self.Up_sample_2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)

        x = self.conv1x1(x)

        out = out + x
        out = self.relu1(out)

        out = self.Up_sample_2(out)

        out = self.attention1(out)

        return out


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


# CBR
class CBR(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out


# ASPP??????
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

        self.att = TripletAttention()

        self.Up_sample_2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.relu3(out)

        x = self.conv1x1(x)
        out = out + x

        out = self.Up_sample_2(out)

        out = self.att(out)

        return out


class refine_module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(refine_module, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.en_block = En_Block(64, 64)

        self.de_block1 = De_Block(64, 64)
        self.de_block2 = De_Block(64, 64)

        self.CBR = CBR(128, 64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        encoders = [out]
        for i in range(4):
            encoder = self.en_block(encoders[i])
            encoders.append(encoder)

        out1 = self.de_block1(encoders[4])
        decoders = [out1]

        for i in range(3):
            decoder = self.de_block2(self.CBR(torch.cat((decoders[i], encoders[-(i + 2)]), dim=1)))
            decoders.append(decoder)

        out = self.conv2(decoders[-1])
        out = self.bn2(out)
        out = self.sigmoid(out)

        return out
