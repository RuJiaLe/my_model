import torch
import torch.nn as nn
from .utils import Video_Decoder_Part
from .utils import CR2, CS, block_aspp_moudle, CS_attention_module, Res_Block
from .ConvGRU import ConvGRUCell
from .resnet_dilation import resnet34


class Video_Encoder_Model(nn.Module):
    def __init__(self, output_stride, input_channels=3, pretrained=True):
        super(Video_Encoder_Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(3, 3), padding=1,
                               stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.resnet = resnet34(pretrained=pretrained)

        self.layer5 = Res_Block(512, 512)

        self.block1_aspp = block_aspp_moudle(in_dim=64, out_dim=64, output_stride=output_stride)
        self.block2_aspp = block_aspp_moudle(in_dim=128, out_dim=128, output_stride=output_stride)
        self.block3_aspp = block_aspp_moudle(in_dim=256, out_dim=256, output_stride=output_stride)
        # self.block4_aspp = block_aspp_moudle(in_dim=512, out_dim=512, output_stride=output_stride)
        # self.block5_aspp = block_aspp_moudle(in_dim=512, out_dim=512, output_stride=output_stride)

        self.CS_attention_module_1 = CS_attention_module(in_channels=64)
        self.CS_attention_module_2 = CS_attention_module(in_channels=128)
        self.CS_attention_module_3 = CS_attention_module(in_channels=256)
        self.CS_attention_module_4 = CS_attention_module(in_channels=512)
        self.CS_attention_module_5 = CS_attention_module(in_channels=512)

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

        block1 = self.resnet.layer1(block0)  # (1, 64, 128, 128)
        block2 = self.resnet.layer2(block1)  # (1, 128, 64, 64)
        block3 = self.resnet.layer3(block2)  # (1, 256, 32, 32)
        block4 = self.resnet.layer4(block3)  # (1, 512, 16, 16)
        block5 = self.layer5(block4)  # (1, 512, 8, 8)

        # print(f'block0: {block0.size()}')
        # print(f'block1: {block1.size()}')
        # print(f'block2: {block2.size()}')
        # print(f'block3: {block3.size()}')
        # print(f'block4: {block4.size()}')
        # print(f'block5: {block5.size()}')

        block1 = self.block1_aspp(block1)
        block2 = self.block2_aspp(block2)
        block3 = self.block3_aspp(block3)
        # block4 = self.block4_aspp(block4)
        # block5 = self.block5_aspp(block5)

        # print(f'block1_result: {block1.size()}')
        # print(f'block2_result: {block2.size()}')
        # print(f'block3_result: {block3_result.size()}')
        # print(f'block4_result: {block4_result.size()}')

        block1 = self.CS_attention_module_1(block1)  # (1, 64, 128, 128)
        block2 = self.CS_attention_module_2(block2)  # (1, 128, 64, 64)
        block3 = self.CS_attention_module_3(block3)  # (1, 256, 32, 32)
        block4 = self.CS_attention_module_4(block4)  # (1, 512, 16, 16)
        block5 = self.CS_attention_module_5(block5)  # (1, 512, 8, 8)

        return [block1, block2, block3, block4, block5]


class Video_Decoder_Model(nn.Module):
    def __init__(self):
        super(Video_Decoder_Model, self).__init__()

        # --------------------第五解码阶段--------------------
        self.decoder5_1 = Video_Decoder_Part(512, 512)
        self.decoder5_2 = Video_Decoder_Part(512, 512)
        self.decoder5_3 = Video_Decoder_Part(512, 512)
        self.decoder5_4 = Video_Decoder_Part(512, 512)

        # --------------------第四解码阶段--------------------
        self.CR4_1 = CR2(1024, 512)
        self.CR4_2 = CR2(1024, 512)
        self.CR4_3 = CR2(1024, 512)
        self.CR4_4 = CR2(1024, 512)

        self.decoder4_1 = Video_Decoder_Part(512, 256)
        self.decoder4_2 = Video_Decoder_Part(512, 256)
        self.decoder4_3 = Video_Decoder_Part(512, 256)
        self.decoder4_4 = Video_Decoder_Part(512, 256)

        # --------------------第三解码阶段--------------------
        self.CR3_1 = CR2(512, 256)
        self.CR3_2 = CR2(512, 256)
        self.CR3_3 = CR2(512, 256)
        self.CR3_4 = CR2(512, 256)

        self.decoder3_1 = Video_Decoder_Part(256, 128)
        self.decoder3_2 = Video_Decoder_Part(256, 128)
        self.decoder3_3 = Video_Decoder_Part(256, 128)
        self.decoder3_4 = Video_Decoder_Part(256, 128)

        # --------------------第二解码阶段--------------------
        self.CR2_1 = CR2(256, 128)
        self.CR2_2 = CR2(256, 128)
        self.CR2_3 = CR2(256, 128)
        self.CR2_4 = CR2(256, 128)

        self.decoder2_1 = Video_Decoder_Part(128, 64)
        self.decoder2_2 = Video_Decoder_Part(128, 64)
        self.decoder2_3 = Video_Decoder_Part(128, 64)
        self.decoder2_4 = Video_Decoder_Part(128, 64)

        # --------------------第一解码阶段--------------------
        self.CR1_1 = CR2(128, 64)
        self.CR1_2 = CR2(128, 64)
        self.CR1_3 = CR2(128, 64)
        self.CR1_4 = CR2(128, 64)

        self.decoder1_1 = Video_Decoder_Part(64, 64)
        self.decoder1_2 = Video_Decoder_Part(64, 64)
        self.decoder1_3 = Video_Decoder_Part(64, 64)
        self.decoder1_4 = Video_Decoder_Part(64, 64)

        # --------------------output阶段--------------------
        self.CS5_1 = CS(512, 1)
        self.CS5_2 = CS(512, 1)
        self.CS5_3 = CS(512, 1)
        self.CS5_4 = CS(512, 1)

        self.CS4_1 = CS(256, 1)
        self.CS4_2 = CS(256, 1)
        self.CS4_3 = CS(256, 1)
        self.CS4_4 = CS(256, 1)

        self.CS3_1 = CS(128, 1)
        self.CS3_2 = CS(128, 1)
        self.CS3_3 = CS(128, 1)
        self.CS3_4 = CS(128, 1)

        self.CS2_1 = CS(64, 1)
        self.CS2_2 = CS(64, 1)
        self.CS2_3 = CS(64, 1)
        self.CS2_4 = CS(64, 1)

        self.CS1_1 = CS(64, 1)
        self.CS1_2 = CS(64, 1)
        self.CS1_3 = CS(64, 1)
        self.CS1_4 = CS(64, 1)

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

    def forward(self, block):  # (1, 512, 8, 8)  [block1, block2, block3, block4, block5]
        # --------------------第五解码阶段--------------------
        out5_1 = self.decoder5_1(block[0][4])  # (1, 512, 16, 16)
        out5_2 = self.decoder5_2(block[1][4])
        out5_3 = self.decoder5_3(block[2][4])
        out5_4 = self.decoder5_4(block[3][4])

        # out4_1 = x4_1
        # out4_2 = x4_2 + (x4_2 - x4_1)
        # out4_3 = x4_3 + (x4_3 - x4_2)
        # out4_4 = x4_4 + (x4_4 - x4_3)

        # --------------------第四解码阶段--------------------
        x4_1 = torch.cat((out5_1, block[0][3]), dim=1)  # (1, 1024, 16, 16)
        x4_2 = torch.cat((out5_2, block[1][3]), dim=1)
        x4_3 = torch.cat((out5_3, block[2][3]), dim=1)
        x4_4 = torch.cat((out5_4, block[3][3]), dim=1)

        x4_1 = self.CR4_1(x4_1)  # (1, 512, 16, 16)
        x4_2 = self.CR4_2(x4_2)
        x4_3 = self.CR4_3(x4_3)
        x4_4 = self.CR4_4(x4_4)

        out4_1 = self.decoder4_1(x4_1)  # (1, 256, 32, 32)
        out4_2 = self.decoder4_2(x4_2)
        out4_3 = self.decoder4_3(x4_3)
        out4_4 = self.decoder4_4(x4_4)

        # out3_1 = x3_1
        # out3_2 = x3_2 + (x3_2 - x3_1)
        # out3_3 = x3_3 + (x3_3 - x3_2)
        # out3_4 = x3_4 + (x3_4 - x3_3)

        # --------------------第三解码阶段--------------------
        x3_1 = torch.cat((out4_1, block[0][2]), dim=1)  # (1, 512, 32, 32)
        x3_2 = torch.cat((out4_2, block[1][2]), dim=1)
        x3_3 = torch.cat((out4_3, block[2][2]), dim=1)
        x3_4 = torch.cat((out4_4, block[3][2]), dim=1)

        x3_1 = self.CR3_1(x3_1)  # (1, 256, 32, 32)
        x3_2 = self.CR3_2(x3_2)
        x3_3 = self.CR3_3(x3_3)
        x3_4 = self.CR3_4(x3_4)

        out3_1 = self.decoder3_1(x3_1)  # (1, 128, 64, 64)
        out3_2 = self.decoder3_2(x3_2)
        out3_3 = self.decoder3_3(x3_3)
        out3_4 = self.decoder3_4(x3_4)

        # --------------------第二解码阶段--------------------
        x2_1 = torch.cat((out3_1, block[0][1]), dim=1)  # (1, 256, 64, 64)
        x2_2 = torch.cat((out3_2, block[1][1]), dim=1)
        x2_3 = torch.cat((out3_3, block[2][1]), dim=1)
        x2_4 = torch.cat((out3_4, block[3][1]), dim=1)

        x2_1 = self.CR2_1(x2_1)  # (1, 128, 64, 64)
        x2_2 = self.CR2_2(x2_2)
        x2_3 = self.CR2_3(x2_3)
        x2_4 = self.CR2_4(x2_4)

        out2_1 = self.decoder2_1(x2_1)  # (1, 64, 128, 128)
        out2_2 = self.decoder2_2(x2_2)
        out2_3 = self.decoder2_3(x2_3)
        out2_4 = self.decoder2_4(x2_4)

        # --------------------第一解码阶段--------------------
        x1_1 = torch.cat((out2_1, block[0][0]), dim=1)  # (1, 128, 128, 128)
        x1_2 = torch.cat((out2_2, block[1][0]), dim=1)
        x1_3 = torch.cat((out2_3, block[2][0]), dim=1)
        x1_4 = torch.cat((out2_4, block[3][0]), dim=1)

        x1_1 = self.CR1_1(x1_1)  # (1, 64, 128, 128)
        x1_2 = self.CR1_2(x1_2)
        x1_3 = self.CR1_3(x1_3)
        x1_4 = self.CR1_4(x1_4)

        out1_1 = self.decoder1_1(x1_1)  # (1, 32, 256, 256)
        out1_2 = self.decoder1_2(x1_2)
        out1_3 = self.decoder1_3(x1_3)
        out1_4 = self.decoder1_4(x1_4)

        # --------------------输出阶段--------------------
        output5_1 = self.Up_sample_16(self.CS5_1(out5_1))
        output5_2 = self.Up_sample_16(self.CS5_2(out5_2))
        output5_3 = self.Up_sample_16(self.CS5_3(out5_3))
        output5_4 = self.Up_sample_16(self.CS5_4(out5_4))

        output4_1 = self.Up_sample_8(self.CS4_1(out4_1))
        output4_2 = self.Up_sample_8(self.CS4_2(out4_2))
        output4_3 = self.Up_sample_8(self.CS4_3(out4_3))
        output4_4 = self.Up_sample_8(self.CS4_4(out4_4))

        output3_1 = self.Up_sample_4(self.CS3_1(out3_1))
        output3_2 = self.Up_sample_4(self.CS3_2(out3_2))
        output3_3 = self.Up_sample_4(self.CS3_3(out3_3))
        output3_4 = self.Up_sample_4(self.CS3_4(out3_4))

        output2_1 = self.Up_sample_2(self.CS2_1(out2_1))
        output2_2 = self.Up_sample_2(self.CS2_2(out2_2))
        output2_3 = self.Up_sample_2(self.CS2_3(out2_3))
        output2_4 = self.Up_sample_2(self.CS2_4(out2_4))

        output1_1 = self.CS1_1(out1_1)
        output1_2 = self.CS1_2(out1_2)
        output1_3 = self.CS1_3(out1_3)
        output1_4 = self.CS1_4(out1_4)

        return [output5_1, output5_2, output5_3, output5_4], \
               [output4_1, output4_2, output4_3, output4_4], \
               [output3_1, output3_2, output3_3, output3_4], \
               [output2_1, output2_2, output2_3, output2_4], \
               [output1_1, output1_2, output1_3, output1_4]
