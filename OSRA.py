
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
