# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权

"""H3 音视频潜空间的分段拼接。

H3 的 latent 是 NestedTensor：video [B,24,T,H/16,W/16] + audio [B,32,2,T]。
视频时间轴按 FRAME_PER_TOKEN=(1,4,4,4,4) 折叠，token 边界落在
1/5/9/13/17/18/22/26... 帧上，所以 22 帧正好是一整组 token。
重叠帧一律按**像素帧**填，内部换成 token；音频按 FRAME_RESCALE=5/3
同步换算，两条时间轴才不会越拼越漂。
"""

import copy

import torch

from comfy.nested_tensor import NestedTensor

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0


def _frames_of_tokens(tokens):
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(tokens))


def _tokens_for_frames(frames):
    """不超过 frames 的最大 token 数。"""
    total, tokens = 0, 0
    while total + FRAME_PER_TOKEN[tokens % 5] <= frames:
        total += FRAME_PER_TOKEN[tokens % 5]
        tokens += 1
    return tokens


def _audio_tokens(frames):
    return int(round(frames * FRAME_RESCALE))


def _split(samples):
    """返回 (video[B,24,T,H,W], audio[B,32,2,T] 或 None)。"""
    if getattr(samples, "is_nested", False):
        tensors = samples.unbind()
        if len(tensors) == 2 and tensors[0].dim() == 5 and tensors[1].dim() == 4:
            return tensors[0], tensors[1]
        raise ValueError("不是 H3 音视频潜空间：期望 video[B,24,T,H,W] + audio[B,32,2,T]")
    if samples.dim() == 5:
        return samples, None
    raise ValueError("潜空间维度不对：期望 5 维视频潜空间，实际 %d 维" % samples.dim())


def _ramp(n, device, dtype):
    """n 帧的重叠过渡权重，0→1 的 smoothstep，避免线性淡化的匀速感。"""
    x = torch.linspace(0.0, 1.0, n + 2, device=device, dtype=dtype)[1:-1]
    return x * x * (3.0 - 2.0 * x)


class FxAiH3LatentStitcher:
    """凤希AI - H3 潜空间拼接（分段累加）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "前段潜空间": ("LATENT",),
                "后段潜空间": ("LATENT",),
                "重叠帧": ("INT", {"default": 22, "min": 0, "max": 340, "step": 1,
                                 "tooltip": "按像素帧填。22 帧正好是 H3 一个 token 组（17k+5 网格），不要填成别的倍数"}),
            },
            "optional": {
                "淡化": ("BOOLEAN", {"default": True, "tooltip": "关掉就是硬接，接缝处会有跳变"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "凤希AI/视频"

    def run(self, 前段潜空间, 后段潜空间, 重叠帧, 淡化=True):
        va, aa = _split(前段潜空间["samples"])
        vb, ab = _split(后段潜空间["samples"])

        if va.shape[0] != vb.shape[0]:
            raise ValueError("两段的 batch 不一致：%d vs %d" % (va.shape[0], vb.shape[0]))
        if va.shape[1] != vb.shape[1] or va.shape[3:] != vb.shape[3:]:
            raise ValueError("两段的通道/尺寸不一致：%s vs %s" % (tuple(va.shape), tuple(vb.shape)))
        if (aa is None) != (ab is None):
            raise ValueError("一段是音视频潜空间、另一段不是，拼不起来")

        ta, tb = va.shape[2], vb.shape[2]
        ot = min(_tokens_for_frames(重叠帧), ta, tb)

        if ot > 0 and 淡化:
            w = _ramp(ot, va.device, va.dtype).view(1, 1, ot, 1, 1)
            joint = va[:, :, ta - ot:] * (1.0 - w) + vb[:, :, :ot] * w
            video = torch.cat([va[:, :, :ta - ot], joint, vb[:, :, ot:]], dim=2)
        else:
            video = torch.cat([va, vb], dim=2)

        out = copy.copy(前段潜空间)
        # 拼完的潜空间覆盖多段，原来的 noise_mask 已经对不上长度
        out.pop("noise_mask", None)

        if aa is None:
            out["samples"] = video
        else:
            ao = min(_audio_tokens(_frames_of_tokens(ta - ot)), aa.shape[-1])
            lb = min(_audio_tokens(_frames_of_tokens(ot)), ab.shape[-1])
            m = min(aa.shape[-1] - ao, lb) if (ot > 0 and 淡化) else 0
            if m > 0:
                wa = _ramp(m, aa.device, aa.dtype).view(1, 1, 1, m)
                joint_audio = aa[..., ao:ao + m] * (1.0 - wa) + ab[..., :m] * wa
            else:
                joint_audio = aa[..., ao:ao]
            out["samples"] = NestedTensor((video, torch.cat([aa[..., :ao], joint_audio, ab[..., m:]], dim=-1)))

        print("[H3潜空间拼接] %d + %d token，重叠 %d token（%d 帧）→ %d token"
              % (ta, tb, ot, _frames_of_tokens(ot), video.shape[2]))
        return (out,)
