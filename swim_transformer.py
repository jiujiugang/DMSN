import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.2, proj_drop=0.2):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.5):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,drop_path=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, to_2tuple(self.window_size), num_heads)
        self.drop_path = drop_path or nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * 4)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W
        shortcut = x

        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        x = shortcut + x
        x = x + self.mlp(self.norm1(x))

        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, downsample=None):
        super().__init__()

        if downsample is not None:
            downsample_input_resolution = (input_resolution[0] * 2, input_resolution[1] * 2)
            downsample_dim = dim // 2
            self.downsample = downsample(downsample_input_resolution, dim=downsample_dim)
        else:
            self.downsample = nn.Identity()

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim,
                                 input_resolution=input_resolution,
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depth)
        ])

    def forward(self, x):
        x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96,
                 depths=[1, 1, 1, 1], num_heads=[3, 6, 12, 24], window_size=7):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_drop = nn.Dropout(0.2)

        patches_resolution = self.patch_embed.grid_size
        self.layers = nn.ModuleList()
        self.num_layers = len(depths)

        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                downsample=PatchMerging if (i_layer > 0) else None
            )
            self.layers.append(layer)

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features


class SwinTransformerFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = SwinTransformer(
            img_size=224,
            patch_size=4,
            in_chans=3,
            embed_dim=96,
            depths=[1, 1, 1, 1],
            num_heads=[3, 6, 12, 24],
            window_size=7
        )
        # 通道调整卷积
        self.adjust_convs = nn.ModuleList([
            nn.Conv2d(96, 64, kernel_size=1),  # 第一阶段 96->64 (修正这里)
            nn.Conv2d(192, 128, kernel_size=1),  # 第二阶段 192->128
            nn.Conv2d(384, 256, kernel_size=1),  # 第三阶段 384->256
            nn.Conv2d(768, 512, kernel_size=1)  # 第四阶段 768->512
        ])

    def forward(self, x):
        features = self.swin(x)  # 原始输出 [B,3136,96], [B,784,192], [B,196,384], [B,49,768]

        # 转换为4D并调整通道
        outputs = []
        spatial_sizes = [56, 28, 14, 7]
        for i, feat in enumerate(features):
            B, L, C = feat.shape
            H = W = spatial_sizes[i]
            # 转换为4D [B, C, H, W]
            feat_4d = feat.transpose(1, 2).reshape(B, C, H, W)
            # 调整通道数
            feat_4d = self.adjust_convs[i](feat_4d)
            outputs.append(feat_4d)

        return outputs  # 返回 [64,64,56,56], [64,128,28,28], [64,256,14,14], [64,512,7,7]

"""
class SwinTransformer(nn.Module):#reshap
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_drop = nn.Dropout(0.2)

        patches_resolution = self.patch_embed.grid_size
        self.layers = nn.ModuleList()
        self.num_layers = len(depths)

        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                downsample=PatchMerging if (i_layer > 0) else None
            )
            self.layers.append(layer)

        # 添加特征形状转换参数
        self.output_channels = [128, 256, 512, 1024]  # 各阶段输出通道数
        self.output_sizes = [56, 28, 14, 7]  # 各阶段输出空间尺寸

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        features = []

        for i, layer in enumerate(self.layers):
            x = layer(x)

            # 将特征转换为4D张量 [B, C, H, W]
            B, L, C = x.shape
            if i == 0:  # 第一阶段 (56x56)
                H_out, W_out = self.output_sizes[0], self.output_sizes[0]
                x_4d = x.transpose(1, 2).reshape(B, C, H_out, W_out)
                # 调整通道数到64
                if C != self.output_channels[0]:
                    x_4d = nn.Conv2d(C, self.output_channels[0], kernel_size=1)(x_4d)

            elif i == 1:  # 第二阶段 (28x28)
                H_out, W_out = self.output_sizes[1], self.output_sizes[1]
                x_4d = x.transpose(1, 2).reshape(B, C, H_out, W_out)
                # 调整通道数到128
                if C != self.output_channels[1]:
                    x_4d = nn.Conv2d(C, self.output_channels[1], kernel_size=1)(x_4d)

            elif i == 2:  # 第三阶段 (14x14)
                H_out, W_out = self.output_sizes[2], self.output_sizes[2]
                x_4d = x.transpose(1, 2).reshape(B, C, H_out, W_out)
                # 调整通道数到256
                if C != self.output_channels[2]:
                    x_4d = nn.Conv2d(C, self.output_channels[2], kernel_size=1)(x_4d)

            elif i == 3:  # 第四阶段 (7x7)
                H_out, W_out = self.output_sizes[3], self.output_sizes[3]
                x_4d = x.transpose(1, 2).reshape(B, C, H_out, W_out)
                # 调整通道数到512
                if C != self.output_channels[3]:
                    x_4d = nn.Conv2d(C, self.output_channels[3], kernel_size=1)(x_4d)

            features.append(x_4d)

        return features
"""

if __name__ == '__main__':
    input_tensor = torch.randn(64, 3, 224, 224)  # 假设batch size为64
    model = SwinTransformerFeatures()

    features = model(input_tensor)
    for idx, feature in enumerate(features):
        print(f"Stage {idx + 1} output shape: {feature.shape}")
"""
if __name__ == '__main__':
    input_tensor = torch.randn(1, 3, 224, 224)
    model = SwinTransformer(img_size=224, patch_size=4, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7)
    features = model(input_tensor)
    for idx, feature in enumerate(features):
        print(f"Layer {idx+1} output shape: {feature.shape}")
    print(model)
"""