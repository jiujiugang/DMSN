import torch

import torch.nn as nn
from HFA_Net.EMA import EMA
from HFA_Net.MSB import MixStructureBlock
from timm.models.layers import DropPath, trunc_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F



import torch

from torch import nn
from torch.nn import functional as F

from mmcv.cnn.bricks import ConvModule, build_activation_layer, build_norm_layer


try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmseg = True
except ImportError:
    print("If for semantic segmentation, please install mmsegmentation first")
    has_mmseg = False

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False
from typing import Optional


class Attention(nn.Module):  ### OSRA,模块通过自注意力机制提取全局特征
    def __init__(self, dim,#输入特征的通道数
                 num_heads=1,#注意力头的数量
                 qk_scale=None,#查询和键的缩放系数。若未提供，则默认为 1/√head_dim。
                 attn_drop=0,#注意力分数的丢弃率
                 sr_ratio=1, ):#下采样比率，若 sr_ratio>1，会引入额外的空间下采样卷积操作，减少计算量。
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio#
        self.q = nn.Conv2d(dim, dim, kernel_size=1)#用于进一步处理降采样后的特征。
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1)#用于进一步处理降采样后的特征。
        self.attn_drop = nn.Dropout(attn_drop)
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                ConvModule(dim, dim,
                           kernel_size=sr_ratio + 3,
                           stride=sr_ratio,
                           padding=(sr_ratio + 3) // 2,
                           groups=dim,
                           bias=False,
                           norm_cfg=dict(type='BN2d'),
                           act_cfg=dict(type='GELU')),
                ConvModule(dim, dim,
                           kernel_size=1,
                           groups=dim,
                           bias=False,
                           norm_cfg=dict(type='BN2d'),
                           act_cfg=None, ), )
        else:
            self.sr = nn.Identity()#决定是否下采样
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)#是 3x3 深度可分离卷积，应用于经过 OSR 下采样的特征图 kv 上，以增强局部特征

    def forward(self, x, relative_pos_enc=None):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)#使用 1x1 卷积生成 q，调整形状以分配给各个注意力头，然后进行维度转置，便于后续计算
        kv = self.sr(x)
        kv = self.local_conv(kv) + kv#通过下采样（self.sr）和局部卷积 self.local_conv 处理输入，得到局部增强的 kv。
        k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)#使用 self.kv(kv) 对处理后的 kv 进行 1x1 卷积，并通过 torch.chunk(..., chunks=2, dim=1) 将其分成两部分，分别赋值给 k 和 v。
        k = k.reshape(B, self.num_heads, C // self.num_heads, -1)
        v = v.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)#使用 self.kv 生成 k 和 v，并 reshape 和 transpose 以匹配多头注意力的格式
        attn = (q @ k) * self.scale#使用矩阵乘法计算 q 与 k 的点积，并乘以缩放系数 self.scale。
        if relative_pos_enc is not None:
            if attn.shape[2:] != relative_pos_enc.shape[2:]:
                relative_pos_enc = F.interpolate(relative_pos_enc, size=attn.shape[2:],
                                                 mode='bicubic', align_corners=False)
            attn = attn + relative_pos_enc#若提供了相对位置编码 relative_pos_enc，则根据 attn 的形状调整后相加
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)#应用 softmax 和 Dropout，得到归一化的注意力权重。
        x = (attn @ v).transpose(-1, -2)#计算 attn 与 v 的加权和（自注意力输出）。
        return x.reshape(B, C, H, W)#调整维度并 reshape 成与输入一致的形状 B, C, H, W。
#OSR 部分对应代码中的 self.sr 和 self.local_conv，用于对输入进行降采样。Linear 层用于生成查询、键和值。MHSA 部分实现了多头自注意力机制，用于提取全局特征。

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class MulScaleBlock(nn.Module):#原始
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(MulScaleBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        scale_width = int(planes / 4)

        self.scale_width = scale_width

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)

        self.conv1_2_1 = conv3x3(scale_width, scale_width)
        self.bn1_2_1 = norm_layer(scale_width)
        self.conv1_2_2 = conv3x3(scale_width, scale_width)
        self.bn1_2_2 = norm_layer(scale_width)
        self.conv1_2_3 = conv3x3(scale_width, scale_width)
        self.bn1_2_3 = norm_layer(scale_width)
        self.conv1_2_4 = conv3x3(scale_width, scale_width)
        self.bn1_2_4 = norm_layer(scale_width)
        self.att = EMA(channels=planes, factor=32)
        self.att1 = Attention(dim=planes)
        #self.att1 = MixStructureBlock(dim=planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        

        sp_x = torch.split(out, self.scale_width, 1)
        out_1_1 = self.bn1_2_1(self.conv1_2_1(sp_x[0]))
        out_1_2 = self.bn1_2_2(self.conv1_2_2(self.relu(out_1_1) + sp_x[1]))
        out_1_3 = self.bn1_2_3(self.conv1_2_3(self.relu(out_1_2) + sp_x[2]))
        out_1_4 = self.bn1_2_4(self.conv1_2_4(self.relu(out_1_3) + sp_x[3]))
        
        out_1 = torch.cat([out_1_1, out_1_2, out_1_3, out_1_4], dim=1)
        
        out =self.att1(out_1)
        #out =self.att1(out_1)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        
        return out


"""
class MulScaleBlock(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None):#inplanes 和 planes 定义了输入和输出的通道数
        super(MulScaleBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        scale_width = int(planes / 2)

        self.scale_width = scale_width

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)

        self.conv1_2_1 = conv3x3(scale_width, scale_width)
        self.bn1_2_1 = norm_layer(scale_width)
        self.conv1_2_2 = conv3x3(scale_width, scale_width)
        self.bn1_2_2 = norm_layer(scale_width)
        self.conv_adjust = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.conv1_2_3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_2_3 = nn.BatchNorm2d(64)
        self.conv1_2_4 = conv3x3(scale_width, scale_width)
        self.bn1_2_4 = norm_layer(scale_width)
        #self.att = EMA(channels=planes, factor=32)
        #self.att1 = MixStructureBlock(dim=planes)
        self.connet = Attention(dim=planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        #sp_x1, sp_x2 = torch.split(out, self.scale_width, 1)
        #out1 = self.relu(self.bn1_2_1(self.conv1_2_1(sp_x1)))
        #out2 = self.relu(self.bn1_2_2(self.conv1_2_2(sp_x2)))
        #out = torch.cat([out1, out2], dim=1)
        #print(out.shape)
        #out =self.conv_adjust(out)

        #out = self.relu(self.bn1_2_3)

        #out = torch.sigmoid(out)


        out = self.connet(out)


        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        
        return out


"""

class MSA(nn.Module):

    def __init__(self, block_b=MulScaleBlock, layers=[1, 1, 1, 1], num_classes=3, zero_init_residual=False):
        super(MSA, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block_b, 64, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block_b, 64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block_b, 128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block_b, 256, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.fc_1 = nn.Linear(512, num_classes)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, block_b):
                    nn.init.constant_(m.bn2.weight, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(conv1x1(inplanes, planes, stride), norm_layer(planes))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        of_part = x
        of_part = self.conv1(of_part)
        of_part = self.bn1(of_part)
        of_part = self.relu(of_part)
        of_part = self.maxpool(of_part)
        of_part = self.layer1(of_part)
        x1 = of_part
        #print(x1.size())

        of_part = self.layer2(of_part)
        x2 = of_part

        of_part = self.layer3(of_part)
        x3 = of_part

        of_part = self.layer4(of_part)  #(8,512,14,14)
        x4 = of_part

        return x1,x2,x3,x4
        # return output

    def forward(self, x):
        return self._forward_impl(x)

if __name__ == "__main__":
    block =MulScaleBlock(inplanes=64, planes=64)
    input_tensor = torch.rand(1, 64, 128, 128)  # B C H W
    output_tensor = block(input_tensor)

    print("Input size:", input_tensor.shape)
    print("Output size:", output_tensor.shape)
