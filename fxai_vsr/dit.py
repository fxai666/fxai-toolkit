# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
"""视频超分的 DiT（原创实现）。

主干结构是 Wan2.1-T2V-1.3B，官方权重在它上面加了三样东西：
  1. 自注意力改成 (2,8,8) 三维窗口分块 + 块级稀疏掩码
  2. 流式推理的逐层 KV 缓存（跨时间块接续）
  3. LQ（低清帧）条件以**加性偏置**注入（`x += lq_bias[block]`，只有 block 0）

权重形状（已核对 `diffusion_pytorch_model_streaming_dmd.safetensors`）：
  dim 1536 / 30 层 / ffn 8960 / patch (1,2,2) / 16 通道 / 12 头 / head_dim 128
  text_embedding 4096→1536（umt5 隐层）  time_projection 9216 = 6*1536
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ops


@dataclass
class DiTConfig:
    dim: int = 1536
    in_dim: int = 16
    out_dim: int = 16
    ffn_dim: int = 8960
    text_dim: int = 4096
    freq_dim: int = 256
    eps: float = 1e-6
    patch_size: tuple = (1, 2, 2)
    num_heads: int = 12
    num_layers: int = 30
    window: tuple = (2, 8, 8)
    # 块稀疏参数（默认值与官方推荐一致）
    sparse_ratio: float = 2.0
    kv_ratio: float = 3.0
    local_range: int = 11
    use_sparse: bool = True

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads


class RMSNorm(nn.Module):
    """Wan 用的 RMSNorm：归一化在 fp32 下做，再乘权重。"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y.to(dtype) * self.weight


class SelfAttention(nn.Module):
    """窗口分块 + 块稀疏 + 流式 KV 缓存的自注意力。"""

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        d, h = cfg.dim, cfg.num_heads
        self.cfg = cfg
        self.num_heads = h
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.norm_q = RMSNorm(d, cfg.eps)
        self.norm_k = RMSNorm(d, cfg.eps)
        # 局部掩码按 (h//8, w//8) 尺寸缓存，尺寸变了才重建
        self._local_mask = None
        self._local_shape = None
        self._local_range = None

    def _get_local_mask(self, gh: int, gw: int, lr: int, device):
        if (self._local_mask is None or self._local_shape != (gh, gw)
                or self._local_range != lr):
            self._local_mask = ops.local_window_mask(gh, gw, lr, True, device=device)
            self._local_shape = (gh, gw)
            self._local_range = lr
        return self._local_mask

    def forward(self, x, freqs, f, h, w, topk, kv_len,
                is_stream=False, cache_k=None, cache_v=None, local_range=11):
        cfg = self.cfg
        b, seq, d = x.shape
        win = cfg.window
        if seq != f * h * w:
            raise ValueError(f"序列长度 {seq} 与 (f,h,w)={(f, h, w)} 不符")

        q = ops.apply_rope(self.norm_q(self.q(x)), freqs, self.num_heads)
        k = ops.apply_rope(self.norm_k(self.k(x)), freqs, self.num_heads)
        v = self.v(x)

        q = q.view(b, f, h, w, d)
        k = k.view(b, f, h, w, d)
        v = v.view(b, f, h, w, d)

        qw = ops.Window3D.partition(q, win)
        kw = ops.Window3D.partition(k, win)
        vw = ops.Window3D.partition(v, win)

        seqlen = f // win[0]
        # one_len 必须在拼接缓存**之前**算，否则会随缓存无限增长
        one_len = kw.shape[0] // b // seqlen
        if cache_k is not None and cache_v is not None:
            kw = torch.cat([cache_k, kw], dim=0)
            vw = torch.cat([cache_v, vw], dim=0)

        nq = qw.shape[0] // b
        bs = qw.shape[1]
        nkv = kw.shape[0] // b

        rq = qw.reshape(b, nq * bs, d)
        rk = kw.reshape(b, nkv * bs, d)
        rv = vw.reshape(b, nkv * bs, d)

        if cfg.use_sparse:
            local = self._get_local_mask(h // 8, w // 8, local_range, kw.device)
            mask = ops.draft_block_mask(qw, kw, self.num_heads, seqlen,
                                        topk, local)
            out = ops.sparse_attention(rq, rk, rv, self.num_heads, mask)
        else:
            out = ops.full_attention(rq, rk, rv, self.num_heads)

        # 滑窗保留最近 kv_len 个时间块
        cur_blocks = kw.shape[0]
        if cur_blocks // one_len > kv_len:
            new_k = kw[one_len:]
            new_v = vw[one_len:]
        else:
            new_k = kw
            new_v = vw

        out = out.reshape(b, nq, bs, d)
        out = ops.Window3D.reverse(out.reshape(b * nq, bs, d), win, (f, h, w))
        out = out.reshape(b, f * h * w, d)
        if is_stream:
            return self.o(out), new_k, new_v
        return self.o(out)


class CrossAttention(nn.Module):
    """对固定文本上下文的交叉注意力。上下文 KV 只算一次，之后每步复用。

    该模型是单步模型且忽略真实提示词，所以这里永远喂同一份
    `posi_prompt.pth` 预计算上下文。
    """

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        d = cfg.dim
        self.num_heads = cfg.num_heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.norm_q = RMSNorm(d, cfg.eps)
        self.norm_k = RMSNorm(d, cfg.eps)
        self.cache_k = None
        self.cache_v = None

    @torch.no_grad()
    def init_cache(self, ctx: torch.Tensor):
        self.cache_k = self.norm_k(self.k(ctx))
        self.cache_v = self.v(ctx)

    def clear_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cache_k is None or self.cache_v is None:
            raise RuntimeError("CrossAttention 的 KV 缓存未初始化，先调 init_cache()")
        q = self.norm_q(self.q(x))
        out = ops.full_attention(q, self.cache_k, self.cache_v, self.num_heads)
        return self.o(out)


class DiTBlock(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        d = cfg.dim
        self.self_attn = SelfAttention(cfg)
        self.cross_attn = CrossAttention(cfg)
        self.norm1 = nn.LayerNorm(d, eps=cfg.eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d, eps=cfg.eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(d, eps=cfg.eps)
        self.ffn = nn.Sequential(
            nn.Linear(d, cfg.ffn_dim), nn.GELU(approximate="tanh"),
            nn.Linear(cfg.ffn_dim, d))
        self.modulation = nn.Parameter(torch.randn(1, 6, d) / d ** 0.5)

    def forward(self, x, t_mod, freqs, f, h, w, topk, kv_len,
                is_stream=False, cache_k=None, cache_v=None, local_range=11):
        m = self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = m.chunk(6, dim=1)

        attn, ck, cv = self.self_attn(
            ops.modulate(self.norm1(x), shift_a, scale_a), freqs, f, h, w,
            topk, kv_len, is_stream=is_stream, cache_k=cache_k,
            cache_v=cache_v, local_range=local_range)
        x = x + gate_a * attn
        x = x + self.cross_attn(self.norm3(x))
        x = x + gate_m * self.ffn(ops.modulate(self.norm2(x), shift_m, scale_m))
        if is_stream:
            return x, ck, cv
        return x


class Head(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        d = cfg.dim
        self.patch_size = cfg.patch_size
        self.norm = nn.LayerNorm(d, eps=cfg.eps, elementwise_affine=False)
        self.head = nn.Linear(d, cfg.out_dim * math.prod(cfg.patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, d) / d ** 0.5)

    def forward(self, x, t):
        m = self.modulation.to(dtype=t.dtype, device=t.device) + t
        shift, scale = m.chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + scale) + shift)


class StreamDiT(nn.Module):
    """完整的 DiT。只前向，不含采样器。"""

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim

        self.patch_embedding = nn.Conv3d(
            cfg.in_dim, d, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(cfg.text_dim, d), nn.GELU(approximate="tanh"),
            nn.Linear(d, d))
        self.time_embedding = nn.Sequential(
            nn.Linear(cfg.freq_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(d, d * 6))
        self.blocks = nn.ModuleList([DiTBlock(cfg) for _ in range(cfg.num_layers)])
        self.head = Head(cfg)

        rf, rh, rw = ops.rope_tables(cfg.head_dim)
        # 注意：不能 register_buffer —— `.to(device, dtype)` 会把 complex128
        # 的虚部直接丢掉（"Casting complex values to real"），旋转就废了。
        # 这里存成普通属性，`.to()` 不会碰它们，用到时再按需搬到卡上。
        self.rope_f, self.rope_h, self.rope_w = rf, rh, rw
        self._freq_cache = {}

    def _freqs(self, f: int, h: int, w: int, t0: int, device):
        """按 (f,h,w,t0) 缓存铺好的频率表，避免每步重做 H2D。"""
        key = (f, h, w, t0, str(device))
        hit = self._freq_cache.get(key)
        if hit is None:
            hit = ops.build_rope_freqs(
                (self.rope_f, self.rope_h, self.rope_w), f, h, w, t0=t0,
                device=device)
            if len(self._freq_cache) > 32:
                self._freq_cache.clear()
            self._freq_cache[key] = hit
        return hit

    # ---------------------------------------------------------------- 上下文
    @torch.no_grad()
    def set_context(self, ctx: torch.Tensor):
        """把固定文本上下文灌进每层的 cross-attn KV 缓存。"""
        emb = self.text_embedding(ctx)
        for blk in self.blocks:
            blk.cross_attn.init_cache(emb)

    def clear_context(self):
        for blk in self.blocks:
            blk.cross_attn.clear_cache()

    # ---------------------------------------------------------------- 时间步
    def time_cond(self, timestep: torch.Tensor):
        """返回 (t, t_mod)。单步模型固定 1000，可只算一次。"""
        t = self.time_embedding(ops.sinusoidal_1d(self.cfg.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.cfg.dim))
        return t, t_mod

    # ---------------------------------------------------------------- 形状
    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid = x.shape[2:]
        return x.flatten(2).transpose(1, 2).contiguous(), grid

    def unpatchify(self, x: torch.Tensor, grid):
        f, h, w = grid
        pf, ph, pw = self.cfg.patch_size
        x = x.view(x.shape[0], f, h, w, pf, ph, pw, self.cfg.out_dim)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        return x.view(x.shape[0], self.cfg.out_dim, f * pf, h * ph, w * pw)

    # ---------------------------------------------------------------- 前向
    def forward(self, x, t, t_mod, lq_bias: Optional[List[torch.Tensor]] = None,
                cur_idx: int = 0, is_stream=False,
                cache_k: Optional[list] = None, cache_v: Optional[list] = None,
                topk_ratio: Optional[float] = None, kv_len: Optional[int] = None,
                local_range: Optional[int] = None):
        """单次前向。

        x: (B, 16, F_lat, H/8, W/8)
        lq_bias: 每层的加性条件，实际只有第 0 层非空
        cache_k/cache_v: 逐层的 KV 缓存列表，流式时原地更新
        返回 (预测, cache_k, cache_v)
        """
        cfg = self.cfg
        win = cfg.window

        x, (f, h, w) = self.patchify(x)
        b = x.shape[0]

        seqlen = f // win[0]
        window_size = win[0] * h * w // 128
        square_num = window_size * window_size
        if topk_ratio is None:
            topk_ratio = cfg.sparse_ratio
        if kv_len is None:
            kv_len = int(cfg.kv_ratio)
        if local_range is None:
            local_range = cfg.local_range
        topk = int(square_num * topk_ratio) - 1

        # 时间位置：流式后续块要接着上一块往后数
        t0 = 0 if cur_idx == 0 else 4 + cur_idx * 2
        freqs = self._freqs(f, h, w, t0, x.device)

        for i, blk in enumerate(self.blocks):
            if lq_bias is not None and i < len(lq_bias):
                x = x + lq_bias[i]
            ck = cache_k[i] if cache_k is not None else None
            cv = cache_v[i] if cache_v is not None else None
            out = blk(x, t_mod, freqs, f, h, w, topk, kv_len,
                      is_stream=is_stream, cache_k=ck, cache_v=cv,
                      local_range=local_range)
            if is_stream:
                x, nk, nv = out
                if cache_k is not None:
                    cache_k[i] = nk
                    cache_v[i] = nv
            else:
                x = out

        x = self.head(x, t)
        return self.unpatchify(x, (f, h, w)), cache_k, cache_v
