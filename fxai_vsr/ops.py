# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
"""视频超分基础算子（原创实现）。

对照的是官方权重体现出的算法行为，不是任何人的源码 —— 结构按 Wan2.1-1.3B 的公开
设计与权重张量形状重建。唯一引入的第三方件是 `third_party/sparse_sage`
（Apache-2.0，SageAttention 团队），见 `third_party/README.md`。

这里放的是与「模型」无关的纯函数：
  · 时间步的正弦嵌入、AdaLN 的 modulate
  · 3D RoPE（时间轴 / 高 / 宽 三向拼接头维）
  · (2,8,8) 窗口分块与还原
  · 局部窗口掩码、块级 topk 稀疏掩码
  · 稀疏注意力调用（含 dense 回退，用于数值验证）
"""

import math

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 时间步嵌入 / AdaLN
# ---------------------------------------------------------------------------


def sinusoidal_1d(dim: int, pos: torch.Tensor) -> torch.Tensor:
    """Transformer 风格的正弦位置嵌入，前半 cos 后半 sin。

    官方权重的 `time_embedding` 输入是 256 维，所以 dim 取 256。用 float64
    算再降回原 dtype —— 低精度下 `10000 ** (-i/n)` 的指数会掉精度。
    """
    half = dim // 2
    idx = torch.arange(half, dtype=torch.float64, device=pos.device)
    freqs = torch.pow(10000.0, -idx / half)
    ang = torch.outer(pos.to(torch.float64), freqs)
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=1).to(pos.dtype)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-zero 的调制：x * (1 + scale) + shift。"""
    return x * (1.0 + scale) + shift


# ---------------------------------------------------------------------------
# 3D RoPE
# ---------------------------------------------------------------------------


def _rope_axis(dim: int, end: int, theta: float) -> torch.Tensor:
    """单个轴的复数旋转因子，形状 (end, dim//2)。"""
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].double() / dim))
    ang = torch.outer(torch.arange(end, device=inv.device), inv)
    return torch.polar(torch.ones_like(ang), ang)


def rope_tables(head_dim: int, end: int = 1024, theta: float = 10000.0):
    """三轴频率表 (freq_f, freq_h, freq_w)。

    头维被切成三段：时间轴拿 `head_dim - 2*(head_dim//3)`，高宽各拿
    `head_dim//3`。head_dim=128 时是 44 / 42 / 42。
    """
    df = head_dim - 2 * (head_dim // 3)
    dh = head_dim // 3
    return _rope_axis(df, end, theta), _rope_axis(dh, end, theta), _rope_axis(dh, end, theta)


def build_rope_freqs(tables, f: int, h: int, w: int, t0: int = 0, device=None):
    """把三轴频率铺成 (f*h*w, 1, head_dim//2) 的复数张量。

    `t0` 是时间轴起点 —— 流式推理时每段要接着上一段的时间位置往下走。
    """
    freq_f, freq_h, freq_w = tables
    ff = freq_f[t0:t0 + f].view(f, 1, 1, -1).expand(f, h, w, -1)
    fh = freq_h[:h].view(1, h, 1, -1).expand(f, h, w, -1)
    fw = freq_w[:w].view(1, 1, w, -1).expand(f, h, w, -1)
    out = torch.cat([ff, fh, fw], dim=-1).reshape(f * h * w, 1, -1)
    return out.to(device) if device is not None else out


def apply_rope(x: torch.Tensor, freqs: torch.Tensor, heads: int) -> torch.Tensor:
    """对 (B, S, heads*head_dim) 施加旋转位置编码。"""
    b, s, _ = x.shape
    x = x.view(b, s, heads, -1)
    # 复数乘法在 float64 下做，避免 bf16 的相位抖动
    xc = torch.view_as_complex(x.to(torch.float64).reshape(b, s, heads, -1, 2))
    xc = xc * freqs.view(1, s, 1, -1)
    return torch.view_as_real(xc).flatten(2).to(x.dtype)


# ---------------------------------------------------------------------------
# 3D 窗口分块
# ---------------------------------------------------------------------------


class Window3D:
    """把 (B, F, H, W, C) 按 (wf, wh, ww) 切成窗口内的长序列，以及逆操作。

    分块的意义：注意力只在窗口内做，序列长度从 F*H*W 降到 wf*wh*ww。
    逆操作必须严格还原原顺序，否则潜空间会错位。
    """

    @staticmethod
    def partition(x: torch.Tensor, win) -> torch.Tensor:
        b, f, h, w, c = x.shape
        wf, wh, ww = win
        if f % wf or h % wh or w % ww:
            raise ValueError(f"尺寸 {(f, h, w)} 不能被窗口 {win} 整除")
        x = x.view(b, f // wf, wf, h // wh, wh, w // ww, ww, c)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        return x.view(-1, wf * wh * ww, c)

    @staticmethod
    def reverse(tokens: torch.Tensor, win, orig) -> torch.Tensor:
        f, h, w = orig
        wf, wh, ww = win
        nf, nh, nw = f // wf, h // wh, w // ww
        b = tokens.shape[0] // (nf * nh * nw)
        x = tokens.view(b, nf, nh, nw, wf, wh, ww, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        return x.view(b, f, h, w, -1)


# ---------------------------------------------------------------------------
# 掩码
# ---------------------------------------------------------------------------


def local_window_mask(gh: int, gw: int, win: int, include_self: bool = True,
                      device=None) -> torch.Tensor:
    """块网格 (gh, gw) 上的滑窗邻接掩码，形状 (gh*gw, gh*gw)。

    与「夹紧」版本的区别：贴边的块窗口会被截断，而不是平移进图内 —— 这样
    每个块看到的邻域是它真实的局部邻域，不会因为贴边而突然多看一块。
    """
    r = torch.arange(gh, device=device)
    c = torch.arange(gw, device=device)
    yy, xx = torch.meshgrid(r, c, indexing="ij")
    r_all = yy.reshape(-1)
    c_all = xx.reshape(-1)
    half = win // 2
    sr = r_all - half
    er = sr + win - 1
    sc = c_all - half
    ec = sc + win - 1
    in_r = (r_all[None, :] >= sr[:, None]) & (r_all[None, :] <= er[:, None])
    in_c = (c_all[None, :] >= sc[:, None]) & (c_all[None, :] <= ec[:, None])
    mask = in_r & in_c
    if not include_self:
        mask.fill_diagonal_(False)
    return mask


@torch.no_grad()
def draft_block_mask(q_win: torch.Tensor, k_win: torch.Tensor, heads: int,
                     seqlen: int, topk: int, local_mask: torch.Tensor) -> torch.Tensor:
    """块级稀疏掩码：先算低分辨率的近似注意力，每行留 topk 个 KV 块。

    思路（与官方实现一致）：把一个窗口内的 128 个 token 平均成 1 个「块向量」
    去打分，注意力分布和真实分布高度相关，但代价只有 1/128。按这个近似分布
    取 topk 个 KV 块，真实注意力就只在这些块上算。

    q_win / k_win: (B*nblk, 128, C)。返回 (B, heads, nq_blk, nkv_sub) 的 bool，
    其中 nkv_sub = nblk*2 —— 打分时 KV 侧被细分成两半（每半 64 token）。

    注意 softmax 必须走 float32：topk 的阈值卡在近似并列的位置，bf16 下
    软概率的末位抖动会让少数元素翻边，掩码就和参考实现对不上。
    """
    b = 1
    block_s = q_win.shape[1]
    sub = 2
    sub_len = block_s // sub

    nq = q_win.shape[0]
    nkv = k_win.shape[0]
    d_head = q_win.shape[2] // heads

    # Q：整块平均
    qa = q_win.mean(dim=1).view(nq, heads, d_head).permute(1, 0, 2)
    # K：拆成两半分别平均，列顺序是「块间交错」——b0h0,b0h1,b1h0,…
    ka = k_win.view(nkv, sub, sub_len, -1).mean(dim=2).reshape(nkv * sub, -1)
    ka = ka.view(nkv * sub, heads, d_head).permute(1, 0, 2)

    # 分两段各算一次，再拼列 —— 与参考实现的算术顺序一致
    k1, k2 = torch.chunk(ka, 2, dim=1)
    scale = math.sqrt(d_head)
    scores = torch.cat([torch.einsum("hld,hmd->hlm", qa, k1) / scale,
                        torch.einsum("hld,hmd->hlm", qa, k2) / scale], dim=-1)

    repeat_len = scores.shape[1] // local_mask.shape[0]
    repeat_num = (scores.shape[2] // sub) // local_mask.shape[1]

    lm = local_mask.unsqueeze(1).unsqueeze(0).repeat(repeat_len, 1, repeat_num, 1)
    # (repeat_len, gh*gw, repeat_num, gw*gw) → (nq, nkv//2)：把前两维拼成行、
    # 后两维拼成列。直接 reshape 就够，行主序天然就是 (x a) / (y b) 的顺序。
    lm = lm.reshape(repeat_len * local_mask.shape[0],
                    repeat_num * local_mask.shape[1])
    # 每个 KV 块被细分成两半，所以列要「逐个复制两遍」而不是整段重复
    lm = lm.repeat_interleave(sub, dim=1)
    lm = lm.unsqueeze(0).repeat(scores.shape[0], 1, 1)

    if lm.shape != scores.shape:
        raise RuntimeError(f"掩码形状 {tuple(lm.shape)} 与打分 {tuple(scores.shape)} 不匹配")

    # 掩码加到 float32 上再做 softmax，避免 bf16 末位抖动影响 topk 取舍
    bias = torch.zeros(scores.shape, dtype=torch.float32, device=scores.device)
    bias = bias.masked_fill(~lm, -float("inf"))
    attn = torch.softmax(scores.float() + bias, dim=-1)

    # 按时间窗分组后取 topk：先展平成 (heads*seqlen, s1*nkv) 再逐行取阈值
    heads, nq, nkv = attn.shape
    per = nq // seqlen
    attn = attn.reshape(heads, seqlen, per, nkv)
    flat = attn.reshape(heads * seqlen, -1)
    n_keep = min(flat.shape[1] - 1, int(topk))
    if n_keep <= 0:
        keep = torch.zeros_like(flat, dtype=torch.bool)
    else:
        thr = torch.topk(flat, k=n_keep + 1, dim=1, largest=True).values[:, -1:]
        keep = flat > thr
    # (heads, seqlen, per, nkv) → (heads, seqlen*per, nkv) = (heads, nq, nkv)
    keep = keep.view(heads, seqlen, per, nkv).reshape(heads, seqlen * per, nkv)
    return keep.unsqueeze(0).repeat(b, 1, 1, 1)


# ---------------------------------------------------------------------------
# 注意力
# ---------------------------------------------------------------------------

_SPARSE_OK = None


def _sparse_available() -> bool:
    global _SPARSE_OK
    if _SPARSE_OK is None:
        try:
            from .third_party.sparse_sage.core import sparse_sageattn  # noqa: F401
            import triton  # noqa: F401
            _SPARSE_OK = True
        except Exception as e:  # pragma: no cover
            print(f"[fxai_vsr] 稀疏注意力不可用，回退 dense：{e}")
            _SPARSE_OK = False
    return _SPARSE_OK


def _to_bh_sd(x: torch.Tensor, heads: int) -> torch.Tensor:
    b, s, _ = x.shape
    return x.view(b, s, heads, -1).permute(0, 2, 1, 3)


def _from_bh_sd(x: torch.Tensor) -> torch.Tensor:
    b, h, s, d = x.shape
    return x.permute(0, 2, 1, 3).reshape(b, s, h * d)


def sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     heads: int, block_mask: torch.Tensor) -> torch.Tensor:
    """块稀疏注意力。q/k/v 为 (B, S, heads*head_dim)，block_mask 为块级 bool。"""
    qh = _to_bh_sd(q, heads)
    kh = _to_bh_sd(k, heads)
    vh = _to_bh_sd(v, heads)
    if _sparse_available():
        from .third_party.sparse_sage.core import sparse_sageattn
        out = sparse_sageattn(qh, kh, vh, mask_id=block_mask.to(torch.int8),
                              is_causal=False, tensor_layout="HND")
        return _from_bh_sd(out)
    return _from_bh_sd(dense_attention(qh, kh, vh, block_mask))


def dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    block_mask: torch.Tensor, q_blk: int = 128,
                    kv_blk: int = 64) -> torch.Tensor:
    """按同一套块掩码做 dense 注意力，用来在没有 Triton 时兜底 / 做数值验证。

    q/k/v 为 (B, heads, S, d)。掩码会从块级展开到 token 级；只在 S 不大时用，
    token 级掩码是 O(S²) 的内存。
    """
    b, h, s, d = q.shape
    s_kv = k.shape[2]
    m = block_mask.repeat_interleave(q_blk, dim=-2).repeat_interleave(kv_blk, dim=-1)
    m = m[:, :, :s, :s_kv].to(torch.bool)
    bias = torch.zeros(b, h, s, s_kv, dtype=q.dtype, device=q.device)
    bias = bias.masked_fill(~m, float("-inf"))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)


def full_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   heads: int) -> torch.Tensor:
    """无掩码全注意力，走 SDPA（会用 flash 后端）。"""
    qh = _to_bh_sd(q, heads)
    kh = _to_bh_sd(k, heads)
    vh = _to_bh_sd(v, heads)
    return _from_bh_sd(F.scaled_dot_product_attention(qh, kh, vh))
