import torch
import torch.nn as nn
from .utils import Video_Decoder_Part
from .utils import CBS, CBR, refine_module
from .ConvGRU import ConvGRUCell
from .resnet_dilation import resnet50
from .triplet_attention import TripletAttention


class Video_Encoder_Model(nn.Module):
    def __init__(self, output_stride, input_channels=3, pretrained=True):
        super(Video_Encoder_Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(3, 3), padding=1,
                               stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.resnet = resnet50(pretrained=pretrained)

        self.block1_attention = TripletAttention()
        self.block2_attention = TripletAttention()
        self.block3_attention = TripletAttention()
        self.block4_attention = TripletAttention()

        self.attention_module = TripletAttention()

        if pretrained:
            for key in self.state_dict():
                if 'resnet' not in key:
                    self.init_layer(key)

    def init_layer(self, key):
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                if self.state_dict()[key].ndimension() >= 2:
                    nn.init.kaiming_normal_(self.state_dict()[key], mode='fan_out', nonlinearity='relu')
            elif 'bn' in key:
                self.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            self.state_dict()[key][...] = 0.001

    def freeze_bn(self):
        for m in self.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eval()

    def forward(self, x):  # (1, 3, 256, 256)
        block0 = self.conv1(x)  # (1, 64, 256, 256)
        block0 = self.bn1(block0)
        block0 = self.relu1(block0)
        block0 = self.max_pool1(block0)  # (1, 64, 128, 128)

        block1_attention = self.block1_attention(block0)
        block1 = self.resnet.layer1(block1_attention)  # (1, 256, 64, 64)

        block2_attention = self.block2_attention(block1)
        block2 = self.resnet.layer2(block2_attention)  # (1, 512, 32, 32)

        block3_attention = self.block3_attention(block2)
        block3 = self.resnet.layer3(block3_attention)  # (1, 1024, 16, 16)

        block4_attention = self.block4_attention(block3)
        block4 = self.resnet.layer4(block4_attention)  # (1, 1024, 8, 8)

        # print(f'block0: {block0.size()}')
        # print(f'block1: {block1.size()}')
        # print(f'block2: {block2.size()}')
        # print(f'block3: {block3.size()}')
        # print(f'block4: {block4.size()}')

        blocks = [block1, block2, block3, block4]
        for i in range(4):
            blocks[i] = self.attention_module(blocks[i])

        return blocks


class Video_Decoder_Model(nn.Module):
    def __init__(self):
        super(Video_Decoder_Model, self).__init__()
        # --------------------第四解码阶段--------------------

        self.decoder4 = Video_Decoder_Part(2048, 1024)

        self.ConvGRU4 = ConvGRUCell(1024, 1024)

        # --------------------第三解码阶段--------------------

        self.CBR3 = CBR(2048, 1024)

        self.decoder3 = Video_Decoder_Part(1024, 512)

        self.ConvGRU3 = ConvGRUCell(512, 512)

        # --------------------第二解码阶段--------------------
        self.CBR2 = CBR(1024, 512)

        self.decoder2 = Video_Decoder_Part(512, 256)

        self.ConvGRU2 = ConvGRUCell(256, 256, kernel_size=3)

        # --------------------第一解码阶段--------------------
        self.CBR1 = CBR(512, 256)

        self.decoder1 = Video_Decoder_Part(256, 128)

        self.ConvGRU1 = ConvGRUCell(128, 128, kernel_size=3)

        # --------------------refine--------------------
        self.refine = refine_module(1, 1)

        # --------------------output阶段--------------------
        self.CBS4 = CBS(1024, 1)

        self.CBS3 = CBS(512, 1)

        self.CBS2 = CBS(256, 1)

        self.CBS1 = CBS(128, 1)

        # --------------------上采样阶段--------------------
        self.Up_sample_16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.Up_sample_8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.Up_sample_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.Up_sample_2 = nn.Upsample(scale_factor=2, mode='bilinear')

    # freeze_bn
    def freeze_bn(self):
        for m in self.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eval()

    def forward(self, block):  # (1, 2048, 8, 8)  [block1, block2, block3, block4]
        # --------------------第四解码阶段--------------------
        x4 = []
        for i in range(len(block)):
            x = self.decoder4(block[i][3])
            x4.append(x)

        out4 = [None]
        for i in range(len(block)):
            out = self.ConvGRU4(x4[i], out4[i])
            out4.append(out)

        out4 = out4[1:]

        # --------------------第三解码阶段--------------------
        x3 = []
        for i in range(len(block)):
            x = torch.cat((x4[i], block[i][2]), dim=1)
            x = self.CBR3(x)
            x = self.decoder3(x)
            x3.append(x)

        out3 = [None]
        for i in range(len(block)):
            out = self.ConvGRU3(x3[i], out3[i])
            out3.append(out)

        out3 = out3[1:]

        # --------------------第二解码阶段--------------------
        x2 = []
        for i in range(len(block)):
            x = torch.cat((x3[i], block[i][1]), dim=1)
            x = self.CBR2(x)
            x = self.decoder2(x)
            x2.append(x)

        out2 = [None]
        for i in range(len(block)):
            out = self.ConvGRU2(x2[i], out2[i])
            out2.append(out)

        out2 = out2[1:]

        # --------------------第一解码阶段--------------------
        x1 = []
        for i in range(len(block)):
            x = torch.cat((out2[i], block[i][0]), dim=1)
            x = self.CBR1(x)
            x = self.decoder1(x)
            x1.append(x)

        out1 = [None]
        for i in range(len(block)):
            out = self.ConvGRU1(x1[i], out1[i])
            out1.append(out)

        out1 = out1[1:]

        # --------------------输出阶段--------------------
        output4 = []
        for i in range(len(block)):
            out = self.Up_sample_16(self.CBS4(out4[i]))
            output4.append(out)

        output3 = []
        for i in range(len(block)):
            out = self.Up_sample_8(self.CBS3(out3[i]))
            output3.append(out)

        output2 = []
        for i in range(len(block)):
            out = self.Up_sample_4(self.CBS2(out2[i]))
            output2.append(out)

        output1 = []
        for i in range(len(block)):
            out = self.Up_sample_2(self.CBS1(out1[i]))
            output1.append(out)

        # --------------------refine--------------------
        output0 = []
        for i in range(len(block)):
            output0.append(self.refine(output1[i]))

        return output4, output3, output2, output1, output0
