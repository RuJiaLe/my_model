import torch
import torch.nn as nn
from .utils import Video_Encoder_Part, self_attention, Video_Decoder_Part
from .utils import CR4, CR2, CS
from .ConvLSTM import ConvLSTMCell


class Video_Encoder_Model(nn.Module):
    def __init__(self, output_stride, input_channels=12, pretrained=True):
        super(Video_Encoder_Model, self).__init__()

        self.self_attention = self_attention()
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
        for m in self.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eval()

    def forward(self, frames):

        out = self.self_attention(frames)
        block = self.encoder(out)

        return block


class Video_Decoder_Model(nn.Module):
    def __init__(self):
        super(Video_Decoder_Model, self).__init__()

        # --------------------第四解码阶段--------------------
        self.decoder4_1 = Video_Decoder_Part(1024, 1024)
        self.decoder4_2 = Video_Decoder_Part(1024, 1024)
        self.decoder4_3 = Video_Decoder_Part(1024, 1024)
        self.decoder4_4 = Video_Decoder_Part(1024, 1024)

        # --------------------第三解码阶段--------------------
        self.CR3_1 = CR2(2048, 1024)
        self.CR3_2 = CR2(2048, 1024)
        self.CR3_3 = CR2(2048, 1024)
        self.CR3_4 = CR2(2048, 1024)

        self.decoder3_1 = Video_Decoder_Part(1024, 512)
        self.decoder3_2 = Video_Decoder_Part(1024, 512)
        self.decoder3_3 = Video_Decoder_Part(1024, 512)
        self.decoder3_4 = Video_Decoder_Part(1024, 512)

        # --------------------第二解码阶段--------------------
        self.CR2_1 = CR2(1024, 512)
        self.CR2_2 = CR2(1024, 512)
        self.CR2_3 = CR2(1024, 512)
        self.CR2_4 = CR2(1024, 512)

        self.decoder2_1 = Video_Decoder_Part(512, 256)
        self.decoder2_2 = Video_Decoder_Part(512, 256)
        self.decoder2_3 = Video_Decoder_Part(512, 256)
        self.decoder2_4 = Video_Decoder_Part(512, 256)

        # --------------------第一解码阶段--------------------
        self.CR1_1 = CR2(512, 256)
        self.CR1_2 = CR2(512, 256)
        self.CR1_3 = CR2(512, 256)
        self.CR1_4 = CR2(512, 256)

        self.decoder1_1 = Video_Decoder_Part(256, 128)
        self.decoder1_2 = Video_Decoder_Part(256, 128)
        self.decoder1_3 = Video_Decoder_Part(256, 128)
        self.decoder1_4 = Video_Decoder_Part(256, 128)

        # --------------------refine--------------------

        self.decoder0_1 = Video_Decoder_Part(128, 64)
        self.decoder0_2 = Video_Decoder_Part(128, 64)
        self.decoder0_3 = Video_Decoder_Part(128, 64)
        self.decoder0_4 = Video_Decoder_Part(128, 64)

        # --------------------output阶段--------------------
        self.CS4_1 = CS(1024, 1)
        self.CS4_2 = CS(1024, 1)
        self.CS4_3 = CS(1024, 1)
        self.CS4_4 = CS(1024, 1)

        self.CS3_1 = CS(512, 1)
        self.CS3_2 = CS(512, 1)
        self.CS3_3 = CS(512, 1)
        self.CS3_4 = CS(512, 1)

        self.CS2_1 = CS(256, 1)
        self.CS2_2 = CS(256, 1)
        self.CS2_3 = CS(256, 1)
        self.CS2_4 = CS(256, 1)

        self.CS1_1 = CS(128, 1)
        self.CS1_2 = CS(128, 1)
        self.CS1_3 = CS(128, 1)
        self.CS1_4 = CS(128, 1)

        self.CS0_1 = CS(64, 1)
        self.CS0_2 = CS(64, 1)
        self.CS0_3 = CS(64, 1)
        self.CS0_4 = CS(64, 1)

        # --------------------上采样阶段--------------------
        self.Up_sample_16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.Up_sample_8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.Up_sample_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.Up_sample_2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def freeze_bn(self):
        for m in self.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eval()

    def forward(self, block):  # (1, 2048, 8, 8)
        # --------------------第四解码阶段--------------------
        out4_1 = self.decoder4_1(block[3])  # (1, 1024, 16, 16)
        out4_2 = self.decoder4_2(block[3])
        out4_3 = self.decoder4_3(block[3])
        out4_4 = self.decoder4_4(block[3])

        # --------------------第三解码阶段--------------------
        x3_1 = torch.cat((out4_1, block[2]), dim=1)
        x3_2 = torch.cat((out4_2, block[2]), dim=1)
        x3_3 = torch.cat((out4_3, block[2]), dim=1)
        x3_4 = torch.cat((out4_4, block[2]), dim=1)

        x3_1 = self.CR3_1(x3_1)
        x3_2 = self.CR3_2(x3_2)
        x3_3 = self.CR3_3(x3_3)
        x3_4 = self.CR3_4(x3_4)

        out3_1 = self.decoder3_1(x3_1)  # (1, 512, 32, 32)
        out3_2 = self.decoder3_2(x3_2)
        out3_3 = self.decoder3_3(x3_3)
        out3_4 = self.decoder3_4(x3_4)

        # --------------------第二解码阶段--------------------
        x2_1 = torch.cat((out3_1, block[1]), dim=1)
        x2_2 = torch.cat((out3_2, block[1]), dim=1)
        x2_3 = torch.cat((out3_3, block[1]), dim=1)
        x2_4 = torch.cat((out3_4, block[1]), dim=1)

        x2_1 = self.CR2_1(x2_1)
        x2_2 = self.CR2_2(x2_2)
        x2_3 = self.CR2_3(x2_3)
        x2_4 = self.CR2_4(x2_4)

        out2_1 = self.decoder2_1(x2_1)  # (1, 256, 64, 64)
        out2_2 = self.decoder2_2(x2_2)
        out2_3 = self.decoder2_3(x2_3)
        out2_4 = self.decoder2_4(x2_4)

        # --------------------第一解码阶段--------------------
        x1_1 = torch.cat((out2_1, block[0]), dim=1)
        x1_2 = torch.cat((out2_2, block[0]), dim=1)
        x1_3 = torch.cat((out2_3, block[0]), dim=1)
        x1_4 = torch.cat((out2_4, block[0]), dim=1)

        x1_1 = self.CR1_1(x1_1)
        x1_2 = self.CR1_2(x1_2)
        x1_3 = self.CR1_3(x1_3)
        x1_4 = self.CR1_4(x1_4)

        out1_1 = self.decoder1_1(x1_1)  # (1, 128, 128, 128)
        out1_2 = self.decoder1_2(x1_2)
        out1_3 = self.decoder1_3(x1_3)
        out1_4 = self.decoder1_4(x1_4)

        # --------------------refine--------------------

        out0_1 = self.decoder0_1(out1_1)  # (1, 64, 256, 256)
        out0_2 = self.decoder0_2(out1_2)
        out0_3 = self.decoder0_3(out1_3)
        out0_4 = self.decoder0_4(out1_4)

        # --------------------输出阶段--------------------
        output4_1 = self.Up_sample_16(self.CS4_1(out4_1))
        output4_2 = self.Up_sample_16(self.CS4_2(out4_2))
        output4_3 = self.Up_sample_16(self.CS4_3(out4_3))
        output4_4 = self.Up_sample_16(self.CS4_4(out4_4))

        output3_1 = self.Up_sample_8(self.CS3_1(out3_1))
        output3_2 = self.Up_sample_8(self.CS3_2(out3_2))
        output3_3 = self.Up_sample_8(self.CS3_3(out3_3))
        output3_4 = self.Up_sample_8(self.CS3_4(out3_4))

        output2_1 = self.Up_sample_4(self.CS2_1(out2_1))
        output2_2 = self.Up_sample_4(self.CS2_2(out2_2))
        output2_3 = self.Up_sample_4(self.CS2_3(out2_3))
        output2_4 = self.Up_sample_4(self.CS2_4(out2_4))

        output1_1 = self.Up_sample_2(self.CS1_1(out1_1))
        output1_2 = self.Up_sample_2(self.CS1_2(out1_2))
        output1_3 = self.Up_sample_2(self.CS1_3(out1_3))
        output1_4 = self.Up_sample_2(self.CS1_4(out1_4))

        output0_1 = self.CS0_1(out0_1)
        output0_2 = self.CS0_2(out0_2)
        output0_3 = self.CS0_3(out0_3)
        output0_4 = self.CS0_4(out0_4)

        return [output4_1, output4_2, output4_3, output4_4], \
               [output3_1, output3_2, output3_3, output3_4], \
               [output2_1, output2_2, output2_3, output2_4], \
               [output1_1, output1_2, output1_3, output1_4], \
               [output0_1, output0_2, output0_3, output0_4]
