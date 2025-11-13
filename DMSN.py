
from HFA_Net.mul_block import MSA
from module.MLFA import MLFA
import torch.nn as nn
import torch
from timm.models.layers import trunc_normal_
from module.AAFF import AAFF
from model.swim_transformer import SwinTransformerFeatures
from model.Swim_mona import SwinTransformer_mona_features
import torch.nn.functional as F
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape

        # 线性变换并分头
        q = self.q(q).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(k).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(v).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 合并头
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, q, kv):
        q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv)))
        return q


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


"""
class DualInception(nn.Module):#双向的原来4阶段
    def __init__(self, num_classes=4,ecg_weight=0.5, optical_weight=0.5,):
        super(DualInception, self).__init__()
        self.msa = MSA()
        self.vis = repvit_m0_6()
        self.swim = SwinTransformerFeatures()
        self.swimmona = SwinTransformer_mona_features()

        # 为每个阶段添加交叉注意力块
        self.cross_attn1 = CrossAttentionBlock(dim=64, num_heads=4)
        self.cross_attn2 = CrossAttentionBlock(dim=128, num_heads=8)
        self.cross_attn3 = CrossAttentionBlock(dim=256, num_heads=16)
        self.cross_attn4 = CrossAttentionBlock(dim=512, num_heads=16)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlfa = MLFA(inter_dim=1024, level=3, channel=[128, 256, 512, 1024])
        self.aaff = AAFF(1024, 512)

        self.head = nn.Linear(512, num_classes) if num_classes > 0 else nn.Identity()
        # 添加分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)  # 假设融合后特征维度为512
        )

        # 初始化损失函数


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    def forward(self, x, y):
        # 获取特征
        m1, m2, m3, m4 = self.msa(y)
        t1, t2, t3, t4 = self.swimmona(x)

        # 定义形状转换函数
        def to_seq(tensor):
            B, C, H, W = tensor.shape
            return tensor.reshape(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)

        def to_spatial(seq, original_shape):
            B, C, H, W = original_shape
            return seq.permute(0, 2, 1).reshape(B, C, H, W)

        # 处理各阶段
        # 阶段1: 64维
        m1_seq, t1_seq = to_seq(m1), to_seq(t1)
        m1_attn = self.cross_attn1(m1_seq, t1_seq)
        t1_attn = self.cross_attn1(t1_seq, m1_seq)
        m1, t1 = to_spatial(m1_attn, m1.shape), to_spatial(t1_attn, t1.shape)

        # 阶段2: 128维
        m2_seq, t2_seq = to_seq(m2), to_seq(t2)
        m2_attn = self.cross_attn2(m2_seq, t2_seq)
        t2_attn = self.cross_attn2(t2_seq, m2_seq)
        m2, t2 = to_spatial(m2_attn, m2.shape), to_spatial(t2_attn, t2.shape)

        # 阶段3: 256维
        m3_seq, t3_seq = to_seq(m3), to_seq(t3)
        m3_attn = self.cross_attn3(m3_seq, t3_seq)
        t3_attn = self.cross_attn3(t3_seq, m3_seq)
        m3, t3 = to_spatial(m3_attn, m3.shape), to_spatial(t3_attn, t3.shape)

        # 阶段4: 512维
        m4_seq, t4_seq = to_seq(m4), to_seq(t4)
        m4_attn = self.cross_attn4(m4_seq, t4_seq)
        t4_attn = self.cross_attn4(t4_seq, m4_seq)
        m4, t4 = to_spatial(m4_attn, m4.shape), to_spatial(t4_attn, t4.shape)

        # 特征拼接和后续处理...
        mt1 = torch.cat((m1, t1), dim=1)
        mt2 = torch.cat((m2, t2), dim=1)
        mt3 = torch.cat((m3, t3), dim=1)
        mt4 = torch.cat((m4, t4), dim=1)

        mt = self.mlfa(mt1, mt2, mt3,mt4)
        output = self.aaff(mt4, mt)#mt4不变，
        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        main_output = self.head(output)
        #print(main_output.shape)

        m4_pooled = self.head(self.avgpool(m4).flatten(1))  # 输出大小: (batch_size, 512)
        t4_pooled = self.head(self.avgpool(t4).flatten(1))  # 输出大小: (batch_size, 512)



        return main_output, m4_pooled, t4_pooled
"""


class DualInception(nn.Module):  # 双向的
    def __init__(self, num_classes=4, ecg_weight=0.5, optical_weight=0.5, ):
        super(DualInception, self).__init__()
        self.msa = MSA()
        # self.vis = repvit_m0_6()
        self.swim = SwinTransformerFeatures()
        self.swimmona = SwinTransformer_mona_features()

        # 为每个阶段添加交叉注意力块
        self.cross_attn1 = CrossAttentionBlock(dim=64, num_heads=4)
        self.cross_attn2 = CrossAttentionBlock(dim=128, num_heads=8)
        self.cross_attn3 = CrossAttentionBlock(dim=256, num_heads=16)
        self.cross_attn4 = CrossAttentionBlock(dim=512, num_heads=16)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlfa = MLFA(inter_dim=1024, level=3, channel=[128, 256, 512, 1024])
        self.aaff = AAFF(1024, 512)

        self.head = nn.Linear(512, num_classes) if num_classes > 0 else nn.Identity()
        # 添加分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)  # 假设融合后特征维度为512
        )

        # 初始化损失函数

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    def forward(self, x, y):
        # 获取特征
        m1, m2, m3, m4 = self.msa(y)
        t1, t2, t3, t4 = self.swimmona(x)

        # 定义形状转换函数
        def to_seq(tensor):
            B, C, H, W = tensor.shape
            return tensor.reshape(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)

        def to_spatial(seq, original_shape):
            B, C, H, W = original_shape
            return seq.permute(0, 2, 1).reshape(B, C, H, W)

        # 处理各阶段
        # 阶段1: 64维
        m1_seq, t1_seq = to_seq(m1), to_seq(t1)
        m1_attn = self.cross_attn1(m1_seq, t1_seq)
        t1_attn = self.cross_attn1(t1_seq, m1_seq)
        m1, t1 = to_spatial(m1_attn, m1.shape), to_spatial(t1_attn, t1.shape)
        """
        # 阶段2: 128维
        m2_seq, t2_seq = to_seq(m2), to_seq(t2)
        m2_attn = self.cross_attn2(m2_seq, t2_seq)
        t2_attn = self.cross_attn2(t2_seq, m2_seq)
        m2, t2 = to_spatial(m2_attn, m2.shape), to_spatial(t2_attn, t2.shape)

        # 阶段3: 256维
        m3_seq, t3_seq = to_seq(m3), to_seq(t3)
        m3_attn = self.cross_attn3(m3_seq, t3_seq)
        t3_attn = self.cross_attn3(t3_seq, m3_seq)
        m3, t3 = to_spatial(m3_attn, m3.shape), to_spatial(t3_attn, t3.shape)

        # 阶段4: 512维

        m4_seq, t4_seq = to_seq(m4), to_seq(t4)
        m4_attn = self.cross_attn4(m4_seq, t4_seq)
        t4_attn = self.cross_attn4(t4_seq, m4_seq)
        m4, t4 = to_spatial(m4_attn, m4.shape), to_spatial(t4_attn, t4.shape)
"""
        # 特征拼接和后续处理...
        mt1 = torch.cat((m1, t1), dim=1)
        mt2 = torch.cat((m2, t2), dim=1)
        mt3 = torch.cat((m3, t3), dim=1)
        mt4 = torch.cat((m4, t4), dim=1)

        mt = self.mlfa(mt1, )
        output = self.aaff(mt4, mt)  # mt4不变，
        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        main_output = self.head(output)
        # print(main_output.shape)

        m4_pooled = self.head(self.avgpool(m4).flatten(1))  # 输出大小: (batch_size, 512)
        t4_pooled = self.head(self.avgpool(t4).flatten(1))  # 输出大小: (batch_size, 512)

        return main_output, m4_pooled, t4_pooled


if __name__ == "__main__":
    model = DualInception(num_classes=4)
    x = torch.rand(8, 3, 224, 224)  # 光流输入
    y = torch.rand(8, 3, 224, 224)  # ECG输入
    labels = torch.randint(0, 4, (8,))  # 假设4分类

    # 训练模式
    output, y, z = model(x, y, labels)
    print(output.shape, y.shape, z.shape)