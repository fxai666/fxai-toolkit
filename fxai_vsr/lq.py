# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
"""LQ（低清帧）条件投影（原创实现）。

作用：把「目标分辨率的低清视频」压成每层一个 1536 维的偏置，加到 DiT 的
token 上。压缩方式：先 16×16 的像素洗牌把空间降 16 倍、通道乘 256，再两个
时间下采样卷积，最后一个线性层出 1536 维。

为什么 LQ 要在**目标分辨率**：模型看到的是已经放大到目标尺寸的低清帧，
它学的是「从这个模糊版还原清晰版」，不是「从低分辨率重建」。所以外面
想放大几倍都行 —— 这也是我们能支持 1.5 倍的原因。

权重（`LQ_proj_in.ckpt`，8 个键，bf16）：
  conv1 (2048, 768, 4,3,3)   norm1.gamma (2048,1,1,1)
  conv2 (3072,2048,4,3,3)    norm2.gamma (3072,1,1,1)
  linear_layers.0 (1536,3072)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE_T = 2


class CausalConv3d(nn.Conv3d):
    """时间维因果的 3D 卷积：只在时间起点补零（用 replicate），终点不补。

    这样第 i 帧的输出只依赖第 ≤i 帧的输入，流式解码才不会看到未来。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pt, ph, pw = self.padding
        # (W_left, W_right, H_top, H_bottom, T_left, T_right)
        self._pad = (pw, pw, ph, ph, 2 * pt, 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache=None):
        pad = list(self._pad)
        if cache is not None and pad[4] > 0:
            x = torch.cat([cache.to(x.device), x], dim=2)
            pad[4] -= cache.shape[2]
        x = F.pad(x, pad, mode="replicate")
        return super().forward(x)


class RMSNormC(nn.Module):
    """按通道做 L2 归一化再乘 sqrt(C)，Wan VAE 用的那一种。"""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma


class PixelShuffle3d(nn.Module):
    """(B, C, F, H, W) → (B, C*ff*hh*ww, F/ff, H/hh, W/ww)。"""

    def __init__(self, ff: int, hh: int, ww: int):
        super().__init__()
        self.ff, self.hh, self.ww = ff, hh, ww

    def forward(self, x):
        b, c, f, h, w = x.shape
        ff, hh, ww = self.ff, self.hh, self.ww
        x = x.reshape(b, c, f // ff, ff, h // hh, hh, w // ww, ww)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
        return x.reshape(b, c * ff * hh * ww, f // ff, h // hh, w // ww)


class LQProjector(nn.Module):
    """把像素域的低清视频投成 DiT 每层的加性条件。"""

    def __init__(self, in_dim: int = 3, out_dim: int = 1536, hidden1: int = 2048,
                 hidden2: int = 3072, layer_num: int = 1):
        super().__init__()
        self.layer_num = layer_num
        self.shuffle = PixelShuffle3d(1, 16, 16)
        self.conv1 = CausalConv3d(in_dim * 256, hidden1, (4, 3, 3),
                                  stride=(2, 1, 1), padding=(1, 1, 1))
        self.norm1 = RMSNormC(hidden1)
        self.conv2 = CausalConv3d(hidden1, hidden2, (4, 3, 3),
                                  stride=(2, 1, 1), padding=(1, 1, 1))
        self.norm2 = RMSNormC(hidden2)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(hidden2, out_dim) for _ in range(layer_num)])
        self.clip_idx = 0

    def reset(self):
        self.clip_idx = 0

    def _heads(self, x):
        """conv2 之后到每层线性投影。x: (B, C, F, H, W) → 每层 (B, F*H*W, 1536)。"""
        tokens = x.flatten(2).transpose(1, 2)
        return [lin(tokens) for lin in self.linear_layers]

    def forward(self, video: torch.Tensor):
        """一次性算完整段。video: (B, 3, F, H, W)，H/W 必须是 16 的倍数。"""
        t = video.shape[2]
        n_iter = 1 + (t - 1) // 4
        # 首帧复制 3 份补在时间起点
        video = torch.cat([video[:, :, :1].repeat(1, 1, 3, 1, 1), video], dim=2)

        pieces = []
        for i in range(n_iter):
            seg = video[:, :, i * 4:(i + 1) * 4]
            x = self.shuffle(seg)
            x = self.conv1(x, x[:, :, -CACHE_T:].clone())
            x = F.silu(self.norm1(x))
            c2 = x[:, :, -CACHE_T:].clone()
            if i == 0:
                continue
            x = self.conv2(x, c2)
            x = F.silu(self.norm2(x))
            pieces.append(x)

        x = torch.cat(pieces, dim=2)
        return self._heads(x)

    def stream_forward(self, clip: torch.Tensor):
        """流式：每次喂一段。第一段只用来建立缓存，返回 None。"""
        if self.clip_idx == 0:
            clip = torch.cat([clip[:, :, :1].repeat(1, 1, 3, 1, 1), clip], dim=2)
            x = self.shuffle(clip)
            x = self.conv1(x, x[:, :, -CACHE_T:].clone())
            F.silu(self.norm1(x))
            self.clip_idx += 1
            return None

        x = self.shuffle(clip)
        x = self.conv1(x, x[:, :, -CACHE_T:].clone())
        x = F.silu(self.norm1(x))
        c2 = x[:, :, -CACHE_T:].clone()
        x = self.conv2(x, c2)
        x = F.silu(self.norm2(x))
        self.clip_idx += 1
        return self._heads(x)
