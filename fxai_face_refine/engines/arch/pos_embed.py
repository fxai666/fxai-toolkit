# 凤希AI：视觉 RoPE（自研实现）
#
# 上游来自 EVA-02 的 rotary embedding（MIT），本实现做了三处调整：
#
# 1. `rotate_half` 用 unflatten + stack + flatten 实现，不走 einops。
#    einops 的 rearrange 每次调用要解析 pattern 字符串、建符号表，
#    而这是 DiT 里层数 × 头数级别的高频调用。语义与上游等价
#    （相邻两个分量互换、前者取负）。
# 2. `VisionRotaryEmbeddingFast.forward` 不做 `clone().to(dtype)`。
#    模型统一 .to(dtype) 之后 buffer 已是目标 dtype。clone 用于防调用方原地改，
#    这里只做乘法，没有原地写。
# 3. 不加 `@torch.compiler.disable`。本引擎不开 compile。
import torch
from einops import repeat
from torch import nn


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(len(t.shape) for t in tensors)
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(len(set(t[1])) <= 2 for t in expandable_dims), "invalid dimensions for broadcastable concatenation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    # 等价于 einops: "... (d r) -> ... d r", r=2 后 stack((-x2, x1), -1)
    xr = x.unflatten(-1, (-1, 2))
    return torch.stack((-xr[..., 1], xr[..., 0]), dim=-1).flatten(-2)


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(self, dim, pt_seq_len=16, theta=10000):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(pt_seq_len) / pt_seq_len * pt_seq_len
        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, t):
        cos, sin = self.freqs_cos, self.freqs_sin
        if cos.dtype != t.dtype:
            cos, sin = cos.to(t.dtype), sin.to(t.dtype)
        return t * cos + rotate_half(t) * sin
