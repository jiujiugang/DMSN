from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()

        # 深度卷积层
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size,
                                                 stride, padding, groups=in_channels, bias=False),
                                       nn.BatchNorm2d(in_channels),
                                       # activation_layer
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
        # 逐点卷积层
        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                       nn.BatchNorm2d(out_channels),
                                       # activation_layer
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x



def Conv(filter_in, filter_out, kernel_size, stride=1, pad=0):
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("silu", nn.SiLU(inplace=True)),
    ]))


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Downsample, self).__init__()

        self.downsample = nn.Sequential(
            Conv(in_channels, out_channels, scale_factor, scale_factor, 0)
        )

    def forward(self, x):
        x = self.downsample(x)

        return x

"""
class MLFA(nn.Module):
    def __init__(self, inter_dim=512, level=4, channel=[64, 128, 256,512],d_model=512, n_heads=8):
        super(MLFA, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_4 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 1, 4, kernel_size=1, stride=1, padding=0)


        self.dsconv = DepthwiseSeparableConv(in_channels=self.inter_dim, out_channels=512)

        self.downsample2x = Downsample(channel[2], channel[3], scale_factor=2)
        self.downsample4x = Downsample(channel[1], channel[3], scale_factor=4)
        self.downsample8x = Downsample(channel[0], channel[3], scale_factor=8)

    def forward(self, x1,):
        input1,  = x1,
        input1 = self.downsample8x(input1)
        #input2 = self.downsample4x(input2)
        #input3 = self.downsample2x(input3)


        level_1_weight_v = self.weight_level_1(input1)
        # print(level_1_weight_v.size())
        #level_2_weight_v = self.weight_level_2(input2)
        # print(level_2_weight_v.size())
        #level_3_weight_v = self.weight_level_3(input3)
        # print(level_3_weight_v.size())
        #level_4_weight_v = self.weight_level_4(input4)#为下采样后的特征图生成对应的权重图
        # print(level_4_weight_v.size())

        levels_weight_v = torch.cat((level_1_weight_v,  ), 1)#通道维度（dim=1）进行拼接，得到一个合并后的权重图   level_2_weight_v, level_3_weight_v,



        levels_weight = self.weight_levels(levels_weight_v)#对拼接后的权重图 levels_weight_v 进行卷积处理，得到一个4通道的权重图 levels_weight。
        levels_weight = F.softmax(levels_weight, dim=1)#对 levels_weight 进行归一化，使得每个尺度的权重总和为1


        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :]
                            #input2 * levels_weight[:, 1:2, :, :] + \
                            #input3 * levels_weight[:, 2:3, :, :] + \
                            #input4 * levels_weight[:, 3:, :, :]#四个尺度的特征图进行加权融合
        #print(fused_out_reduced.shape)



        out = self.dsconv(fused_out_reduced)

        return out


"""
class MLFA(nn.Module):
    def __init__(self, inter_dim=512, level=4, channel=[64, 128, 256,512],d_model=512, n_heads=8):
        super(MLFA, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_4 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)


        self.dsconv = DepthwiseSeparableConv(in_channels=self.inter_dim, out_channels=512)

        self.downsample2x = Downsample(channel[2], channel[3], scale_factor=2)
        self.downsample4x = Downsample(channel[1], channel[3], scale_factor=4)
        self.downsample8x = Downsample(channel[0], channel[3], scale_factor=8)

    def forward(self, x1,x2,x3,x4):
        input1, input2, input3, input4 = x1,x2, x3, x4
        input1 = self.downsample8x(input1)
        input2 = self.downsample4x(input2)
        input3 = self.downsample2x(input3)


        level_1_weight_v = self.weight_level_1(input1)
        # print(level_1_weight_v.size())
        level_2_weight_v = self.weight_level_2(input2)
        # print(level_2_weight_v.size())
        level_3_weight_v = self.weight_level_3(input3)
        # print(level_3_weight_v.size())
        level_4_weight_v = self.weight_level_4(input4)#为下采样后的特征图生成对应的权重图
        # print(level_4_weight_v.size())

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v, level_4_weight_v), 1)#通道维度（dim=1）进行拼接，得到一个合并后的权重图



        levels_weight = self.weight_levels(levels_weight_v)#对拼接后的权重图 levels_weight_v 进行卷积处理，得到一个4通道的权重图 levels_weight。
        levels_weight = F.softmax(levels_weight, dim=1)#对 levels_weight 进行归一化，使得每个尺度的权重总和为1


        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :] + \
                            input3 * levels_weight[:, 2:3, :, :] + \
                            input4 * levels_weight[:, 3:, :, :]#四个尺度的特征图进行加权融合
        #print(fused_out_reduced.shape)



        out = self.dsconv(fused_out_reduced)

        return out
