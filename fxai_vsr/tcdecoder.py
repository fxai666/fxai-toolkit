# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
"""视频超分的潜空间解码器 TCDecoder（原创实现）。

结构是 TAEHV 那一系的「极小自编码器」解码端：三段 [MemBlock×3 → 2×上采样
→ 时间上采样 → 1×1 卷积]，每段通道减半，最后出 3 通道。每段之间用
IdentityConv2d(3×3, Dirac 初始化) 加深一层感受野。

两个关键点：
  1. **MemBlock** 带「上一帧」的记忆，所以解码是严格流式的，跨时间块接续
  2. 解码时把 LQ 帧经过 `PixelShuffle3d(4,8,8)` 拼进通道维（16 + 768 = 784），
     模型据此把细节补回去

权重 `TCDecoder.ckpt`：66 个键，fp32，首层 (512, 784, 3, 3)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

TWorkItem = namedtuple("TWorkItem", ("x", "block"))


def _conv(n_in, n_out, **kw):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kw)


class IdentityConv2d(nn.Conv2d):
    """同尺寸 3×3 卷积，权重按 Dirac 初始化（等价恒等映射）。"""

    def __init__(self, c, kernel_size=3, bias=False):
        super().__init__(c, c, kernel_size, padding=kernel_size // 2, bias=bias)
        with torch.no_grad():
            nn.init.dirac_(self.weight)
            if self.bias is not None:
                self.bias.zero_()


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    """带时间记忆的残差块。`past` 是上一帧的同位置输入。"""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            _conv(n_in * 2, n_out), nn.ReLU(inplace=True),
            _conv(n_out, n_out), nn.ReLU(inplace=True),
            _conv(n_out, n_out),
        )
        self.skip = (nn.Conv2d(n_in, n_out, 1, bias=False)
                     if n_in != n_out else nn.Identity())

    def forward(self, x, past):
        return F.relu(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TPool(nn.Module):
    """时间下采样：stride 帧并成一帧。"""

    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x):
        nt, c, h, w = x.shape
        return self.conv(x.reshape(-1, self.stride * c, h, w))


class TGrow(nn.Module):
    """时间上采样：一帧变 stride 帧。"""

    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        nt, c, h, w = x.shape
        return self.conv(x).reshape(-1, c, h, w)


class CondShuffle(nn.Module):
    """把 LQ 帧 (B,C,F,H,W) 洗成 (B, F/4, C*256, H/8, W/8) 供通道拼接。

    时间不是 4 的倍数时，在最前面补「首帧」——注意是首帧而不是尾帧，
    这样补出来的时间片在语义上属于过去，不会泄漏未来。
    """

    def __init__(self, ff=4, hh=8, ww=8):
        super().__init__()
        self.ff, self.hh, self.ww = ff, hh, ww

    def forward(self, x):
        b, c, f, h, w = x.shape
        ff, hh, ww = self.ff, self.hh, self.ww
        if f % ff:
            pad = ff - f % ff
            x = torch.cat([x[:, :, :1].repeat(1, 1, pad, 1, 1), x], dim=2)
        f2 = x.shape[2]
        x = x.reshape(b, c, f2 // ff, ff, h // hh, hh, w // ww, ww)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
        # → (B, T', C*ff*hh*ww, H/hh, W/ww)，和 latent 一样是 NTCHW，才能按通道拼
        return x.reshape(b, c * ff * hh * ww, f2 // ff, h // hh, w // ww).transpose(1, 2)


class TCDecoder(nn.Module):
    """潜空间 → RGB 的流式解码器。"""

    image_channels = 3

    def __init__(self, latent_channels=784, channels=(512, 256, 128, 128),
                 time_upscale=(True, True), space_upscale=(True, True, True)):
        super().__init__()
        self.latent_channels = latent_channels
        self.frames_to_trim = 2 ** sum(time_upscale) - 1
        c = list(channels)

        base = [
            Clamp(), _conv(latent_channels, c[0]), nn.ReLU(inplace=True),
            MemBlock(c[0], c[0]), MemBlock(c[0], c[0]), MemBlock(c[0], c[0]),
            nn.Upsample(scale_factor=2 if space_upscale[0] else 1),
            TGrow(c[0], 1),
            _conv(c[0], c[1], bias=False),

            MemBlock(c[1], c[1]), MemBlock(c[1], c[1]), MemBlock(c[1], c[1]),
            nn.Upsample(scale_factor=2 if space_upscale[1] else 1),
            TGrow(c[1], 2 if time_upscale[0] else 1),
            _conv(c[1], c[2], bias=False),

            MemBlock(c[2], c[2]), MemBlock(c[2], c[2]), MemBlock(c[2], c[2]),
            nn.Upsample(scale_factor=2 if space_upscale[2] else 1),
            TGrow(c[2], 2 if time_upscale[1] else 1),
            _conv(c[2], c[3], bias=False),

            nn.ReLU(inplace=True),
            _conv(c[3], self.image_channels),
        ]
        self.decoder = nn.Sequential(*self._deepen(base))
        self.cond_shuffle = CondShuffle(4, 8, 8)
        self.mem = [None] * len(self.decoder)

    @staticmethod
    def _deepen(layers):
        """每个 ReLU 后面插一层 IdentityConv2d + ReLU，扩感受野不加参数负担。"""
        out = []
        for b in layers:
            out.append(b)
            if isinstance(b, nn.ReLU):
                ch = None
                if len(out) >= 2 and isinstance(out[-2], nn.Conv2d):
                    ch = out[-2].out_channels
                elif len(out) >= 2 and isinstance(out[-2], MemBlock):
                    ch = out[-2].conv[-1].out_channels
                if ch is not None:
                    out.append(IdentityConv2d(ch, 3, bias=False))
                    out.append(nn.ReLU(inplace=True))
        return out

    # ------------------------------------------------------------------ 流式
    def clean_mem(self):
        self.mem = [None] * len(self.decoder)

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        """NTCHW → NTCHW，逐帧流过整个栈，MemBlock 带跨帧/跨块记忆。"""
        n, t, c, h, w = x.shape
        out = []
        queue = [TWorkItem(xt, 0)
                 for xt in x.reshape(n, t * c, h, w).chunk(t, dim=1)]

        while queue:
            xt, i = queue.pop(0)
            if i == len(self.decoder):
                out.append(xt)
                continue
            blk = self.decoder[i]

            if isinstance(blk, MemBlock):
                if self.mem[i] is None:
                    y = blk(xt, xt * 0)
                    self.mem[i] = xt
                else:
                    y = blk(xt, self.mem[i])
                    self.mem[i].copy_(xt)
                queue.insert(0, TWorkItem(y, i + 1))

            elif isinstance(blk, TPool):
                if self.mem[i] is None:
                    self.mem[i] = []
                self.mem[i].append(xt)
                if len(self.mem[i]) > blk.stride:
                    self.mem[i] = self.mem[i][-blk.stride:]
                elif len(self.mem[i]) == blk.stride:
                    nn_, cc_, hh_, ww_ = xt.shape
                    y = blk(torch.cat(self.mem[i], 1).view(nn_ * blk.stride, cc_, hh_, ww_))
                    self.mem[i] = []
                    queue.insert(0, TWorkItem(y, i + 1))

            elif isinstance(blk, TGrow):
                y = blk(xt)
                nn_, cc_, hh_, ww_ = y.shape
                for piece in reversed(y.view(n, blk.stride * cc_, hh_, ww_)
                                      .chunk(blk.stride, 1)):
                    queue.insert(0, TWorkItem(piece, i + 1))

            else:
                queue.insert(0, TWorkItem(blk(xt), i + 1))

        return torch.stack(out, 1)

    def decode(self, latents: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        """latents: (B, T, 16, h, w)；cond: (B, 3, F, H, W) 的 LQ 帧。

        返回 (B, T', 3, H, W)，范围 [0, 1]。
        """
        trim = self.mem[-8] is None
        if cond is not None:
            latents = torch.cat([self.cond_shuffle(cond), latents], dim=2)
        out = self._run(latents)
        return out[:, self.frames_to_trim:] if trim else out
