
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from timm.models.layers import DropPath, trunc_normal_

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)
class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x
class ChannelAttention(nn.Module):  # 通道注意力机制
    def __init__(self, in_planes, scaling=16):  # scaling为缩放比例，
        # 用来控制两个全连接层中间神经网络神经元的个数，一般设置为16，具体可以根据需要微调
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // scaling, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // scaling, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out


class SpatialAttention(nn.Module):  # 空间注意力机制
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x


class CBAM_Attention(nn.Module):
    def __init__(self, channel, scaling=16, kernel_size=7):
        super(CBAM_Attention, self).__init__()
        self.channelattention = ChannelAttention(channel, scaling=scaling)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class AAFF(nn.Module):

    def __init__(self, input_channels, internal_neurons):#internal_neurons: 内部的通道数缩减到多少
        super(AAFF, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.cov = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=False)
        # self.dropout = nn.Dropout(p=0.1)  # 添加一个dropout层，p表示丢弃概率
        self.cov2 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=False)
        # self.input_channels = input_channels
        self.spatial_attention = SpatialAttention(kernel_size=7)  # 空间注意力机制
        self.channel_attention = ChannelAttention(input_channels, scaling=16)  # 通道注意力机制

    def forward(self, inputs,mt):
        inputs = self.cov2(inputs)
        inputs = torch.cat((inputs, mt), dim=1)
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        # x1 = self.dropout(x1)  # 添加dropout层
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        #x1 = x1 * self.spatial_attention(x1)  # 在x1上应用空间注意力机制
        #x2 = x2 * self.channel_attention(x2)  # 在x2上应用通道注意力机制


        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        # x2 = self.dropout(x2)  # 添加dropout层
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1*inputs + x2*inputs
        # x = x + mt
        x = self.cov2(x)
        return x