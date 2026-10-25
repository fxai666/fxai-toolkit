# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
"""并行 DiTBlock：纯张量并行（TP），每卡都持有一份完整序列。

为什么不是序列并行（SP）
------------------------
SP 的收益是「残差流按 token 切开，每卡只存 1/world」。但那是**训练**的口径 ——
训练要存激活做反向，切了才省得下来。推理是 `inference_mode`，层与层之间不留激活，
残差流只有一份 [S, K]：S=39000、K=5376、bf16 时 400 MiB，切开每卡只省 200 MiB。
为了这 200 MiB 要付的代价是：

- 每个 block 多两次集合通信（all_gather 拿全序列 + all_to_all 换回头组），
  实测 61 + 83 = 144 ms/block，占单层总耗时 12%；
- `out_proj` 没法切（all_to_all 之后每卡已经有全部头，再切只多一次归约），
  50 层 × 36.8 MiB = **1.80 GiB 每卡都要复制一份**；
- `mod_segments` 要按 rank 的 token 范围重裁，`sizes/offs` 要一路传，容易错。

纯 TP 反过来：残差流每卡整份（+200 MiB），但 `out_proj` 能按列切（−0.90 GiB/卡），
而且每 block 只剩两次 all_reduce，没有 gather 也没有 all_to_all。净省 0.7 GiB、
少 144 ms，代码还少一半。

切分方式（都是「按输出行 / 按输入列」，经典 Megatron 那一套）
--------------------------------------------------------------
    qkv_proj   [3*H*D, K]  按行切 → 每卡算自己那 Hl 个头
    out_proj   [K, H*D]    按列切 → 每卡算出自己头组的贡献（部分和）→ all_reduce
    fc1        [2*F, K]    按行切，gate/up 成对
    fc2        [K, F]      按列切 → 部分和 → all_reduce

    norm1/norm2/adaln  不切，每卡一份（体积可以忽略）

q_norm/k_norm 是 per-head RMSNorm、rope 是 per-head 的，按头切天然精确。
swiglu 是 `chunk(2)`，所以 fc1 必须 gate/up 成对切，不能切连续两半。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.quant_ops
from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift
from comfy.ldm.modules.attention import (
    AttentionTensorContainer,
    ComfyAttention,
    optimized_attention,
)

try:
    from comfy_kitchen.tensor import QuantizedTensor
except ImportError:  # pragma: no cover
    QuantizedTensor = ()


__all__ = [
    "slice_weight", "ShardLinear", "ParallelDiTBlock",
    "split_even", "check_split",
]


# --------------------------------------------------------------------- 权重切片
def _is_quantized(w):
    return isinstance(w, QuantizedTensor) if QuantizedTensor != () else False


def slice_weight(weight, rows=None, cols=None):
    """按行 / 按列切一块 Linear 权重，int8 量化张量也支持。

    H3 的 int8 checkpoint 是 `int8_tensorwise + convrot(groupsize=256)`：
    旋转只沿输入维做，输出行彼此独立，所以按行切精确无损；按列切只要切点落在
    256 的整数倍上（7168 = 28×256、14336 = 56×256 都满足）同样精确。
    """
    if not _is_quantized(weight):
        w = weight
        if rows is not None:
            w = w[rows]
        if cols is not None:
            w = w[:, cols]
        return w.contiguous()

    cls = weight.layout_cls
    qdata, scale = cls.get_plain_tensors(weight)[:2]
    p = weight.params
    if rows is not None:
        qdata = qdata[rows]
        scale = scale[rows]
    if cols is not None:
        qdata = qdata[:, cols]
    params = cls.Params(scale=scale.float().contiguous(), orig_dtype=p.orig_dtype,
                        orig_shape=tuple(qdata.shape), is_weight=p.is_weight,
                        convrot=p.convrot, convrot_groupsize=p.convrot_groupsize,
                        transposed=p.transposed)
    return QuantizedTensor(qdata.contiguous(), cls.__name__, params)


class ShardLinear:
    """只带权重的极简 linear。权重可以是普通张量，也可以是 QuantizedTensor。

    QuantizedTensor 走 `F.linear` 时会由 `__torch_dispatch__` 路由到量化 kernel，
    所以不需要 ComfyUI 那套 `cast_bias_weight`。**代价是绕过 `weight_function`
    钩子 —— 并行模式下 LoRA / 权重补丁不生效**，这是已知限制。
    """

    __slots__ = ("weight",)

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x):
        return F.linear(x, self.weight)

    def to(self, device):
        self.weight = self.weight.to(device)
        return self

    def __repr__(self):
        return f"<ShardLinear {tuple(self.weight.shape)}>"


# --------------------------------------------------------------------- 切分校验
def check_split(heads, head_dim, ffn_half, world):
    """能否等分。返回 (ok, 说明)。

    头数按头切（56 / 2 = 28 ✓，56 / 4 = 14 ✓，56 / 3 ✗）；
    ffn 半宽按 256 的倍数切（14336 = 56×256，能被 2/4 整除，不能被 3 整除）。
    除不尽目前直接拒绝 —— 不等分切分还没做，硬切会让 convrot 的 256 组错位。
    """
    head_ok = heads % world == 0
    ffn_ok = ffn_half % world == 0 and (ffn_half // world) % 256 == 0
    detail = (f"头 {heads}/{world}={'整除' if head_ok else '除不尽'}，"
              f"ffn {ffn_half}/{world}="
              f"{'整除' if ffn_ok else '除不尽或不是 256 的倍数'}")
    return head_ok and ffn_ok, detail


def split_even(heads, ffn_half, world):
    """等分方案：每卡几个头、多少 ffn 通道。"""
    ok, detail = check_split(heads, 128, ffn_half, world)
    if not ok:
        raise ValueError(f"{world} 卡切不开：{detail}")
    return heads // world, ffn_half // world


# --------------------------------------------------------------------- 并行 block
class ParallelDiTBlock:
    """把一个 `DiTBlock` 的权重切成 world 份，只保留本 rank 那份。

    不参与切分的（norm / adaln）保持整份，构造时直接引用原模块 —— 调用方要保证
    这些模块在本 rank 的显存上。

    对外签名与 `DiTBlock.forward` 一致（多一个 `mod=`），可以直接顶替进
    `model.blocks`。
    """

    def __init__(self, block, rank, world, exchange, device=None, mlp_chunk=8192):
        heads = block.attn.heads
        head_dim = block.attn.head_dim
        ffn_out = block.mlp.fc1.weight.shape[0]      # 28672 = 2 * 14336
        if ffn_out % 2:
            raise ValueError(f"fc1 输出宽 {ffn_out} 不是偶数，不像 swiglu")
        ffn_half = ffn_out // 2                      # 14336 = gate | up
        heads_local, ffn_local = split_even(heads, ffn_half, world)

        self.rank = rank
        self.world = world
        self.ex = exchange
        self.device = device
        self.heads = heads
        self.head_dim = head_dim
        self.heads_local = heads_local
        self.ffn_local = ffn_local
        self.hidden = block.norm1.weight.shape[0]
        self.mlp_chunk = mlp_chunk

        attn = block.attn
        inner = heads * head_dim                    # 7168
        # qkv：[3*H*D, K]，每卡取 q/k/v 三段里属于自己的那 Hl 个头
        a = rank * heads_local * head_dim
        b = a + heads_local * head_dim
        rows = torch.cat([torch.arange(sec * inner + a, sec * inner + b)
                          for sec in range(3)])
        self.qkv = ShardLinear(slice_weight(attn.qkv_proj.weight, rows=rows))
        # out_proj：[K, H*D]，按输入列切 —— 本卡只算自己头组的贡献
        self.out_proj = ShardLinear(
            slice_weight(attn.out_proj.weight, cols=slice(a, b)))
        self.q_norm = attn.q_norm
        self.k_norm = attn.k_norm
        self.comfy_attention = ComfyAttention()

        mlp = block.mlp
        per = ffn_local
        # gate / up 成对取：swiglu 是 chunk(2)，只取一半会配错
        ffn_rows = torch.cat([
            torch.arange(rank * per, (rank + 1) * per),
            torch.arange(ffn_half + rank * per, ffn_half + (rank + 1) * per),
        ])
        self.fc1 = ShardLinear(slice_weight(mlp.fc1.weight, rows=ffn_rows))
        # fc2 按输入切：本卡的 per 段正好对上 swiglu 之后的宽度
        self.fc2 = ShardLinear(
            slice_weight(mlp.fc2.weight, cols=slice(rank * per, (rank + 1) * per)))

        self.norm1 = block.norm1
        self.norm2 = block.norm2
        self.adaln_proj = block.adaln_proj

        # 切片是从源张量（可能在某张卡上）切出来的，必须搬到本 rank 的卡上
        if device is not None:
            for lin in (self.qkv, self.out_proj, self.fc1, self.fc2):
                lin.to(device)

        # 缓冲 key：同一 rank 内所有 block 依次使用，可以共用一组缓冲
        self.k_attn = "ar_attn"
        self.k_mlp = "ar_mlp"

    # ---------------------------------------------------------------- 内部
    def _attention(self, h, rope_freqs, transformer_options):
        """在完整序列上、只对自家 Hl 个头做 attention，返回 [S, Hl*D]。"""
        s = h.shape[0]
        hl, d = self.heads_local, self.head_dim
        q, k, v = self.qkv(h).split(hl * d, dim=-1)
        v = v.view(s, hl, d)

        if rope_freqs is not None:
            # 与原版一致：per-head RMSNorm + partial split-half rope，原地做在 qkv 上
            q = q.view(1, s, hl, d)
            k = k.view(1, s, hl, d)
            qw = comfy.model_management.cast_to(self.q_norm.weight, device=h.device)
            kw = comfy.model_management.cast_to(self.k_norm.weight, device=h.device)
            rot = rope_freqs.shape[-3] * 2
            if comfy.model_management.in_training:
                q, k = comfy.quant_ops.ck.rms_rope_split_half(
                    q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
            else:
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
            q = q[0]
            k = k[0]
        else:
            q = self.q_norm(q.view(s, hl, d))
            k = self.k_norm(k.view(s, hl, d))

        q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
        k = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
        v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
        out = optimized_attention(q, k, v, hl, preferred_attention=self.comfy_attention,
                                  mask=None, skip_reshape=True,
                                  transformer_options=transformer_options)
        return out.squeeze(0)

    def _mlp(self, h):
        """按 token 分块跑 fc1→swiglu→fc2，返回 [S, hidden] 的**部分和**。

        不分块时中间量 [S, 2*ffn/world] 在 S=39000、两卡时是 1.07 GiB；
        分块之后峰值只跟块大小有关（8192 块 ≈ 224 MiB）。
        """
        s = h.shape[0]
        y = torch.empty((s, self.hidden), dtype=h.dtype, device=h.device)
        for i in range(0, s, self.mlp_chunk):
            yc = self.fc1(h[i:i + self.mlp_chunk])
            g, u = yc.chunk(2, dim=-1)
            y[i:i + self.mlp_chunk] = self.fc2(F.silu(g) * u)
        return y

    # ---------------------------------------------------------------- 前向
    def forward(self, x, t_emb, mod_segments, rope_freqs,
                transformer_options=None, attention=None, mod=None):
        """x: [S, hidden]，每卡一份完整序列；mod_segments 覆盖 [0, S) 原样使用。

        `mod` 是预先算好的 6 个 modulation 向量。adaln 被卸载到 CPU 时，调用方应该
        在 `ex.run` 之外算一次再传进来 —— 两个 rank 用的是同一份 modulation，
        让它们各自调一遍 CpuModule 会白算一次（CPU 上是串行的）。
        """
        transformer_options = transformer_options or {}

        if mod is None:
            mod = self.adaln_proj(t_emb)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod

        # --- attention 段：qkv 切头 → 本头组 attention → out_proj 切列 → 归约
        h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
        a = self.out_proj(self._attention(h, rope_freqs, transformer_options))
        a = self.ex.all_reduce_sum(a, key=self.k_attn, inplace=True)
        x = _mod_gate(x, gate_msa, a, mod_segments)

        # --- mlp 段：fc1 切 ffn → fc2 切列 → 归约
        h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
        y = self.ex.all_reduce_sum(self._mlp(h), key=self.k_mlp, inplace=True)
        return _mod_gate(x, gate_mlp, y, mod_segments)

    __call__ = forward

    def __repr__(self):
        return (f"<ParallelDiTBlock rank{self.rank}/{self.world} "
                f"heads {self.heads_local}/{self.heads} ffn {self.ffn_local}>")
