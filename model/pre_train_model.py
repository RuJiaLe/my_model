import torch
import torch.nn as nn
from .utils import Video_Encoder_Part, pre_train_decoder
from .utils import CBR4, CBR2, CBS
from .ConvLSTM import ConvLSTMCell


class Video_Encoder_Model(nn.Module):
    def __init__(self, output_stride, input_channels=3, pretrained=True):
        super(Video_Encoder_Model, self).__init__()
        self.encoder = Video_Encoder_Part(output_stride=output_stride, input_channels=input_channels,
                                          pretrained=pretrained)

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
        for m in self.backbone.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eval()

    def forward(self, frame):
        block = self.encoder(frame)

        return block


class Video_Decoder_Model(nn.Module):
    def __init__(self):
        super(Video_Decoder_Model, self).__init__()

        # --------------------第四解码阶段--------------------
        self.decoder4 = pre_train_decoder(1024, 512)
        # --------------------第三解码阶段--------------------
        self.CBR3_1 = CBR4(1024, 256)
        self.CBR3_2 = CBR4(1024, 256)
        self.CBR3_3 = CBR4(1024, 256)
        self.CBR3_4 = CBR4(1024, 256)

        self.ConvLSTM3_1 = ConvLSTMCell(256, 256)
        self.ConvLSTM3_2 = ConvLSTMCell(256, 256)
        self.ConvLSTM3_3 = ConvLSTMCell(256, 256)
        self.ConvLSTM3_4 = ConvLSTMCell(256, 256)

        # --------------------第二解码阶段--------------------
        self.CBR2_1 = CBR2(512, 256)
        self.CBR2_2 = CBR2(512, 256)
        self.CBR2_3 = CBR2(512, 256)
        self.CBR2_4 = CBR2(512, 256)

        self.decoder2 = pre_train_decoder(256, 128)

        # --------------------第一解码阶段--------------------
        self.CBR1_1 = CBR4(256, 64)
        self.CBR1_2 = CBR4(256, 64)
        self.CBR1_3 = CBR4(256, 64)
        self.CBR1_4 = CBR4(256, 64)

        self.ConvLSTM1_1 = ConvLSTMCell(64, 64)
        self.ConvLSTM1_2 = ConvLSTMCell(64, 64)
        self.ConvLSTM1_3 = ConvLSTMCell(64, 64)
        self.ConvLSTM1_4 = ConvLSTMCell(64, 64)

        # --------------------output阶段--------------------
        self.CBS4_1 = CBS(512, 1)
        self.CBS4_2 = CBS(512, 1)
        self.CBS4_3 = CBS(512, 1)
        self.CBS4_4 = CBS(512, 1)

        self.CBS3_1 = CBS(256, 1)
        self.CBS3_2 = CBS(256, 1)
        self.CBS3_3 = CBS(256, 1)
        self.CBS3_4 = CBS(256, 1)

        self.CBS2_1 = CBS(128, 1)
        self.CBS2_2 = CBS(128, 1)
        self.CBS2_3 = CBS(128, 1)
        self.CBS2_4 = CBS(128, 1)

        self.CBS1_1 = CBS(64, 1)
        self.CBS1_2 = CBS(64, 1)
        self.CBS1_3 = CBS(64, 1)
        self.CBS1_4 = CBS(64, 1)

        # --------------------上采样阶段--------------------
        self.Up_sample_8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.Up_sample_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.Up_sample_2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def freeze_bn(self):
        for m in self.backbone.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eval()

    def forward(self, blocks):  # (1, 1024, 16, 16)
        # --------------------第四解码阶段--------------------
        x4_1 = blocks[0][3]
        x4_2 = blocks[1][3]
        x4_3 = blocks[2][3]
        x4_4 = blocks[3][3]
        out4_1, out4_2, out4_3, out4_4 = self.decoder4(x4_1, x4_2, x4_3, x4_4)  # (1, 512, 32, 32)

        # --------------------第三解码阶段--------------------
        x3_1 = blocks[0][2]  # (1, 512, 32, 32)
        x3_2 = blocks[1][2]
        x3_3 = blocks[2][2]
        x3_4 = blocks[3][2]

        x3_1 = self.CBR3_1(torch.cat((x3_1, out4_1), dim=1))  # (1, 256, 32, 32)
        x3_2 = self.CBR3_2(torch.cat((x3_2, out4_2), dim=1))
        x3_3 = self.CBR3_3(torch.cat((x3_3, out4_3), dim=1))
        x3_4 = self.CBR3_4(torch.cat((x3_4, out4_4), dim=1))

        out3_1 = self.Up_sample_2(x3_1)  # (1, 256, 64, 64)
        out3_2 = self.Up_sample_2(x3_2)
        out3_3 = self.Up_sample_2(x3_3)
        out3_4 = self.Up_sample_2(x3_4)

        # --------------------第二解码阶段--------------------
        x2_1 = blocks[0][1]  # (1, 256, 64, 64)
        x2_2 = blocks[1][1]
        x2_3 = blocks[2][1]
        x2_4 = blocks[3][1]

        x2_1 = self.CBR2_1(torch.cat((x2_1, out3_1), dim=1))  # (1, 256, 64, 64)
        x2_2 = self.CBR2_2(torch.cat((x2_2, out3_2), dim=1))
        x2_3 = self.CBR2_3(torch.cat((x2_3, out3_3), dim=1))
        x2_4 = self.CBR2_4(torch.cat((x2_4, out3_4), dim=1))

        out2_1, out2_2, out2_3, out2_4 = self.decoder2(x2_1, x2_2, x2_3, x2_4)  # (1, 128, 128, 128)

        # --------------------第一解码阶段--------------------
        x1_1 = blocks[0][0]  # (1, 128, 128, 128)
        x1_2 = blocks[1][0]
        x1_3 = blocks[2][0]
        x1_4 = blocks[3][0]

        x1_1 = self.CBR1_1(torch.cat((x1_1, out2_1), dim=1))  # (1, 256, 32, 32)
        x1_2 = self.CBR1_2(torch.cat((x1_2, out2_2), dim=1))
        x1_3 = self.CBR1_3(torch.cat((x1_3, out2_3), dim=1))
        x1_4 = self.CBR1_4(torch.cat((x1_4, out2_4), dim=1))

        out1_1 = self.Up_sample_2(x1_1)  # (1, 64, 256,256)
        out1_2 = self.Up_sample_2(x1_2)
        out1_3 = self.Up_sample_2(x1_3)
        out1_4 = self.Up_sample_2(x1_4)

        # --------------------输出阶段--------------------
        output4_1 = self.Up_sample_8(self.CBS4_1(out4_1))
        output4_2 = self.Up_sample_8(self.CBS4_2(out4_2))
        output4_3 = self.Up_sample_8(self.CBS4_3(out4_3))
        output4_4 = self.Up_sample_8(self.CBS4_4(out4_4))

        output3_1 = self.Up_sample_4(self.CBS3_1(out3_1))
        output3_2 = self.Up_sample_4(self.CBS3_2(out3_2))
        output3_3 = self.Up_sample_4(self.CBS3_3(out3_3))
        output3_4 = self.Up_sample_4(self.CBS3_4(out3_4))

        output2_1 = self.Up_sample_2(self.CBS2_1(out2_1))
        output2_2 = self.Up_sample_2(self.CBS2_2(out2_2))
        output2_3 = self.Up_sample_2(self.CBS2_3(out2_3))
        output2_4 = self.Up_sample_2(self.CBS2_4(out2_4))

        output1_1 = self.CBS1_1(out1_1)
        output1_2 = self.CBS1_2(out1_2)
        output1_3 = self.CBS1_3(out1_3)
        output1_4 = self.CBS1_4(out1_4)

        return [output4_1, output4_2, output4_3, output4_4], \
               [output3_1, output3_2, output3_3, output3_4], \
               [output2_1, output2_2, output2_3, output2_4], \
               [output1_1, output1_2, output1_3, output1_4]
