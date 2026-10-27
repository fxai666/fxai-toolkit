# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
"""视频超分流式推理管线（原创实现）。

与官方节点的关键差别：
  1. **任意放大倍数**。模型只要求「LQ 已经处在目标分辨率」，所以 1.5 倍、
     2.25 倍都能跑；官方节点把倍数写死成 2 或 4。
  2. **分块按目标分辨率对齐**。目标尺寸必须是 128 的倍数（窗口分块 (2,8,8)
     要求 patchified 网格能被 8 整除），所以分块也按目标空间的 128 倍数切，
     而不是按源像素切。
  3. **外边框不羽化**。分块拼接时只有真有邻居的那一侧才做渐变，否则整帧
     外框会被压成一条黑带。

推理本身是**单步**的：模型把 x 直接预测成 x0（DMD 蒸馏），没有采样循环。
时间维靠流式 KV 缓存接续，每块 2 个潜帧（第一块 6 个）。
"""

import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .loader import build_models

# 目标尺寸必须是 128 的倍数
SIZE_MULT = 128


# ---------------------------------------------------------------------------
# 小波色彩校正：保留生成的细节，换回低清的色调
# ---------------------------------------------------------------------------


def _blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    """3×3 高斯核按 radius 做空洞卷积，等价于小波的低通。"""
    k = torch.tensor([[0.0625, 0.125, 0.0625],
                      [0.125, 0.25, 0.125],
                      [0.0625, 0.125, 0.0625]], dtype=x.dtype, device=x.device)
    n, c, h, w = x.shape
    weight = k.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    xp = F.pad(x, (radius,) * 4, mode="replicate")
    return F.conv2d(xp, weight, stride=1, padding=0, dilation=radius, groups=c)


def _decompose(x: torch.Tensor, levels: int = 5):
    """多尺度分解：返回 (细节层累加, 最底层低频)。"""
    high = torch.zeros_like(x)
    low = x
    for i in range(levels):
        b = _blur(low, 2 ** i)
        high = high + (low - b)
        low = b
    return high, low


def color_fix(hq: torch.Tensor, lq: torch.Tensor, levels: int = 5,
              clip=(-1.0, 1.0)) -> torch.Tensor:
    """把 hq 的高频 + lq 的低频拼起来。

    hq/lq: (B, C, F, H, W)，同一尺寸。这样出来的画面既有重建的细节，
    色调又跟低清原片一致 —— 不会整体偏色或提亮。
    """
    b, c, f, h, w = hq.shape
    hq4 = hq.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
    lq4 = lq.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
    h_high, _ = _decompose(hq4, levels)
    _, l_low = _decompose(lq4, levels)
    out = torch.clamp(h_high + l_low, *clip)
    return out.reshape(b, f, c, h, w).permute(0, 2, 1, 3, 4)


# ---------------------------------------------------------------------------
# 尺寸工具
# ---------------------------------------------------------------------------


def largest_8n1(n: int) -> int:
    """不超过 n 的最大 8k+1。"""
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def align_size(w: int, h: int, mult: int = SIZE_MULT):
    """把目标尺寸取到**最接近**的 mult 倍数。

    取最近而不是向上取整：向上取整会让内部渲染被拉伸。1280×768 放 1.2 倍是
    1536×922，向上取整成 1536×1024 就是纵向多拉了 11%，再从 1024 里裁掉 102 行，
    画面就被压扁了（网格线实测横线间距 213 而竖线 192）。取最近是 1536×896，
    只差 2.8%，回缩到精确尺寸时肉眼看不出来，还省 11% 算力。
    """
    aw = max(mult, int(round(w / mult)) * mult)
    ah = max(mult, int(round(h / mult)) * mult)
    return aw, ah


def tile_coords(h: int, w: int, tile: int, overlap: int):
    """在目标空间按 tile 边长切块，返回 (x1, y1, x2, y2) 列表。"""
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("分块重叠必须小于块边长")
    ny = max(1, math.ceil((h - overlap) / stride))
    nx = max(1, math.ceil((w - overlap) / stride))
    out = []
    for r in range(ny):
        for c in range(nx):
            y1 = min(r * stride, max(0, h - tile))
            x1 = min(c * stride, max(0, w - tile))
            out.append((x1, y1, min(x1 + tile, w), min(y1 + tile, h)))
    return out


def feather_mask(h: int, w: int, ov: int, edges, device, dtype):
    """块内羽化遮罩。edges=(左, 上, 右, 下) 为 True 表示那一侧有邻居。"""
    m = torch.ones(1, 1, h, w, device=device, dtype=dtype)
    if ov <= 0:
        return m
    ramp = torch.linspace(0, 1, ov, device=device, dtype=dtype)
    left, top, right, bottom = edges
    if left:
        m[..., :ov] = torch.minimum(m[..., :ov], ramp.view(1, 1, 1, -1))
    if right:
        m[..., -ov:] = torch.minimum(m[..., -ov:], ramp.flip(0).view(1, 1, 1, -1))
    if top:
        m[..., :ov, :] = torch.minimum(m[..., :ov, :], ramp.view(1, 1, -1, 1))
    if bottom:
        m[..., -ov:, :] = torch.minimum(m[..., -ov:, :], ramp.flip(0).view(1, 1, -1, 1))
    return m


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------


class VideoUpscaler:
    """常驻的视频超分器。权重只加载一次，可反复调用。"""

    def __init__(self, dtype=torch.bfloat16, device="cuda", use_sparse=True,
                 log: Callable = print):
        self.device = device
        self.dtype = dtype
        self.log = log
        self.dit, self.lq, self.dec, self.ctx = build_models(
            dtype=dtype, device=device, use_sparse=use_sparse)
        self.dit.set_context(self.ctx)
        ts = torch.tensor([1000.0], device=device, dtype=dtype)
        self.t, self.t_mod = self.dit.time_cond(ts)
        self.n_blocks = len(self.dit.blocks)

    # ------------------------------------------------------------------ 单块
    @torch.no_grad()
    def _run_block(self, lq: torch.Tensor, seed: int, color: bool,
                   progress: Optional[Callable] = None) -> torch.Tensor:
        """lq: (1, 3, F, Ht, Wt)，已处在目标分辨率、范围 [-1,1]。"""
        _, _, n_frames, ht, wt = lq.shape
        n_align = largest_8n1(n_frames + 4)
        if n_align < 25:
            raise ValueError(
                f"帧数不足：{n_frames} 帧对齐后只有 {n_align} 帧，"
                f"流式管线至少要 21 帧（对齐到 25 帧）才跑得动一个时间块")
        if n_align > n_frames:
            lq = torch.cat([lq, lq[:, :, -1:].repeat(1, 1, n_align - n_frames, 1, 1)],
                           dim=2)

        n_chunks = (n_align - 1) // 8 - 2
        if n_chunks < 1:
            raise ValueError(f"对齐后 {n_align} 帧，不足以跑一个时间块")

        gen = torch.Generator(device=self.device).manual_seed(seed)
        lat = torch.randn((1, 16, (n_align - 1) // 4, ht // 8, wt // 8),
                          generator=gen, device=self.device, dtype=self.dtype)

        self.lq.reset()
        self.dec.clean_mem()
        cache_k = [None] * self.n_blocks
        cache_v = [None] * self.n_blocks
        lq_pre = 0
        pieces = []

        for ci in range(n_chunks):
            if ci == 0:
                bias = None
                for j in range(7):
                    seg = lq[:, :, max(0, j * 4 - 3):(j + 1) * 4 - 3]
                    cur = self.lq.stream_forward(seg)
                    if cur is None:
                        continue
                    bias = cur if bias is None else [
                        torch.cat([a, b], dim=1) for a, b in zip(bias, cur)]
                lq_cur = 21
                cur_lat = lat[:, :, :6]
            else:
                bias = None
                for j in range(2):
                    s = ci * 8 + 17 + j * 4
                    cur = self.lq.stream_forward(lq[:, :, s:s + 4])
                    if cur is None:
                        continue
                    bias = cur if bias is None else [
                        torch.cat([a, b], dim=1) for a, b in zip(bias, cur)]
                lq_cur = ci * 8 + 21
                cur_lat = lat[:, :, 4 + ci * 2:6 + ci * 2]

            pred, cache_k, cache_v = self.dit(
                cur_lat, self.t, self.t_mod, lq_bias=bias, cur_idx=ci,
                is_stream=True, cache_k=cache_k, cache_v=cache_v)
            cur_lat = cur_lat - pred

            cond = lq[:, :, lq_pre:lq_cur]
            # cur_lat 是 BCFHW（DiT 的约定），解码器要 NTCHW
            frames = self.dec.decode(cur_lat.transpose(1, 2), cond=cond)
            frames = frames.transpose(1, 2).mul_(2).sub_(1)
            if color:
                frames = color_fix(frames, cond)
            pieces.append(frames.cpu())
            lq_pre = lq_cur
            if progress:
                progress(ci + 1, n_chunks)

        out = torch.cat(pieces, dim=2)
        return out

    # ------------------------------------------------------------------ 整段
    @torch.no_grad()
    def upscale(self, frames: torch.Tensor, scale: float = 2.0,
                tile: Optional[int] = None, overlap: int = 64, seed: int = 0,
                color: bool = True, progress: Optional[Callable] = None,
                ) -> torch.Tensor:
        """frames: (N, H, W, 3) 或 (N, H, W, C)，范围 [0,1]。

        返回 (M, Ht, Wt, 3)，范围 [0,1]。M 可能比 N 少几帧（流式对齐的固有损失）。
        """
        if frames.dim() != 4:
            raise ValueError(f"需要 (N,H,W,C) 的图片序列，收到 {tuple(frames.shape)}")
        n, h, w, c = frames.shape
        if c < 3:
            raise ValueError("至少需要 3 个通道")

        # 1) 目标尺寸：先取精确值，再上取到 128 的倍数
        want_w, want_h = int(round(w * scale)), int(round(h * scale))
        aw, ah = align_size(want_w, want_h)
        self.log(f"[视频超分] 源 {w}×{h} ×{scale:g} → 目标 {want_w}×{want_h}"
                 f"（按 {SIZE_MULT} 对齐为 {aw}×{ah}）")

        # 2) 逐帧 bicubic 放大到工作尺寸，转 [-1,1]，摆成 (1,3,N,H,W)
        src = frames[..., :3].permute(0, 3, 1, 2).to(self.device, self.dtype)
        big = F.interpolate(src, size=(ah, aw), mode="bicubic",
                            align_corners=False).clamp(0, 1)
        lq_all = big.mul_(2).sub_(1).permute(1, 0, 2, 3).unsqueeze(0)

        # 3) 分块 or 整帧
        if tile is None or (aw <= tile and ah <= tile):
            out = self._run_block(lq_all, seed, color, progress)
        else:
            out = self._run_tiled(lq_all, tile, overlap, seed, color, progress)

        # 4) 回到 [0,1]，缩放回精确目标尺寸
        #    注意是**缩放**不是裁切：工作尺寸跟目标尺寸差的那点必须整体缩回去，
        #    裁掉多余部分等于把画面纵向压扁（1.2 倍时差 11%，肉眼一眼能看出来）。
        out = out.squeeze(0).permute(1, 2, 3, 0).float().add_(1).mul_(0.5).clamp_(0, 1)
        if (aw, ah) != (want_w, want_h):
            out = F.interpolate(out.permute(0, 3, 1, 2), size=(want_h, want_w),
                                mode="bicubic", align_corners=False)
            out = out.clamp_(0, 1).permute(0, 2, 3, 1)
        return out

    # ------------------------------------------------------------------ 分块
    @torch.no_grad()
    def _run_tiled(self, lq: torch.Tensor, tile: int, overlap: int, seed: int,
                   color: bool, progress: Optional[Callable] = None) -> torch.Tensor:
        """按目标空间切块分别推理，再按羽化权重拼回去。

        画布放在 **CPU**：它跟整段帧数 × 目标尺寸成正比（1920×1152 / 36 帧就是
        1 GB 级别），留在显存里会把本来够用的余量吃掉。拼完再统一搬回去。
        """
        _, c, n, ht, wt = lq.shape
        coords = tile_coords(ht, wt, tile, overlap)
        self.log(f"[视频超分] 分块 {tile}（重叠 {overlap}）→ {len(coords)} 块")

        cpu = torch.device("cpu")
        canvas = None
        weight = None

        for i, (x1, y1, x2, y2) in enumerate(coords):
            if progress:
                progress(i, len(coords))
            sub = lq[:, :, :, y1:y2, x1:x2]
            piece = self._run_block(sub, seed + i * 7919, color)
            _, pc, pn, ph, pw = piece.shape

            if canvas is None:
                # 流式对齐会吃掉几帧，画布按实际输出帧数开
                canvas = torch.zeros(1, pc, pn, ht, wt, dtype=torch.float32)
                weight = torch.zeros(1, 1, pn, ht, wt, dtype=torch.float32)

            ov = min(overlap, ph // 2, pw // 2)
            m1 = feather_mask(ph, pw, ov,
                              (x1 > 0, y1 > 0, x2 < wt, y2 < ht),
                              cpu, torch.float32).unsqueeze(2)
            canvas[:, :, :, y1:y1 + ph, x1:x1 + pw] += piece * m1
            weight[:, :, :, y1:y1 + ph, x1:x1 + pw] += m1

        return canvas / weight.clamp_min(1e-8)
