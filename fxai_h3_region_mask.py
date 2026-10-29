# -*- coding: utf-8 -*-
"""H3 区域蒙版：把像素空间的人物蒙版注入音视频联合潜空间的 noise_mask。

为什么需要它
------------
H3 的 ref2v 只有"参考"能力，没有"局部编辑"能力。给一段源视频 + 一张参考图，
模型往往不是替换指定的人，而是在画面里**新增**一个参考角色，原人物继续演——
因为 conditioning 里没有"只准改这块"的约束。

官方 Relay 2.0 用一个 SAM 蒙版 + ApplyEditMask 解决这件事。本节点做同一件事，
但不依赖任何第三方包：把像素蒙版缩到 latent 网格，写成 ComfyUI 采样器认的
noise_mask，采样时只有蒙版内的位置被去噪重绘，其余位置保持源视频原样。

音轨侧必须留零
--------------
H3 用 audio 侧的 mask 判断"音轨是保持的"，口型同步靠它驱动。本节点只写 video
侧，audio 侧沿用上游（如 NativeAudioLock / H3PerFrameDenoise）已有的蒙版，
没有就写全零，绝不能动。
"""

import torch
import torch.nn.functional as F

import comfy.nested_tensor


def _grow(mask, radius):
    """二值蒙版膨胀（radius 为像素，0 表示不动）。"""
    if radius <= 0:
        return mask
    k = radius * 2 + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)


def _feather(mask, radius):
    """羽化：连续做几次平均池化，近似高斯，避免蒙版边缘出现硬切。"""
    if radius <= 0:
        return mask
    out = mask
    for _ in range(3):
        out = F.avg_pool2d(out, kernel_size=radius * 2 + 1, stride=1, padding=radius)
    return out


class FxAiH3RegionMask:
    """把像素蒙版写进 H3 音视频潜空间，采样时只重绘蒙版内区域。"""

    CATEGORY = "凤希AI/视频"
    FUNCTION = "run"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("潜空间", "报告")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": ("LATENT", {
                    "tooltip": "MiniMax H3 的音视频联合潜空间（ref2v 的 LATENT 输出）"}),
                "mask": ("MASK", {
                    "tooltip": "像素空间蒙版：要重绘（换成参考角色）的区域。多帧蒙版按时间轴对齐"}),
                "扩展": ("INT", {"default": 16, "min": -256, "max": 512, "step": 1,
                                 "tooltip": "蒙版向外扩张的像素数，用来把人物边缘/头发也包进来。负数表示收缩"}),
                "羽化": ("INT", {"default": 8, "min": 0, "max": 256, "step": 1,
                                 "tooltip": "蒙版边缘渐变像素数，避免接缝处出现硬边"}),
                "反选": ("BOOLEAN", {"default": False,
                                     "tooltip": "勾上则重绘蒙版之外的区域（例如只保住人物、重画背景）"}),
            }
        }

    def run(self, av_latent, mask, 扩展, 羽化, 反选):
        samples = av_latent.get("samples")
        if not (isinstance(samples, comfy.nested_tensor.NestedTensor)
                or getattr(samples, "is_nested", False)):
            raise ValueError("FxAiH3RegionMask 需要 MiniMax H3 的音视频联合潜空间（NestedTensor）")

        members = list(samples.unbind())
        video = members[0]                       # [B, 24, T, H, W]
        if video.ndim != 5:
            raise ValueError(f"video 潜空间维度异常：{tuple(video.shape)}")

        B, C, T_lat, H_lat, W_lat = video.shape

        # ---- 蒙版规整到 [1, 1, T, H, W]（latent 网格）----
        m = mask
        if m.ndim == 2:
            m = m.unsqueeze(0)
        m = m.float()
        if m.ndim != 3:
            m = m.reshape(-1, m.shape[-2], m.shape[-1])

        t_px, h_px, w_px = m.shape
        m = m.unsqueeze(1)                       # [T, 1, H, W]

        # 空间：像素网格 -> latent 网格
        if (h_px, w_px) != (H_lat, W_lat):
            m = F.interpolate(m, size=(H_lat, W_lat), mode="bilinear",
                              align_corners=False)

        # 时间：像素帧 -> latent 帧。
        # 这里必须"分组取最大值"而不是线性插值：一个 latent 帧覆盖 4 个像素帧
        # （首帧单独），只要这 4 帧里有人物，这一格就必须重绘；插值会把人物
        # 进出的边界糊成半透明，结果既没完全换、也没完全保，出现重影。
        if t_px != T_lat:
            if t_px == 1:
                m = m.expand(T_lat, -1, -1, -1)
            else:
                m5 = m.permute(1, 0, 2, 3).unsqueeze(0)   # [1,1,T,H,W]
                m5 = F.adaptive_max_pool3d(m5, (T_lat, H_lat, W_lat))
                m = m5.squeeze(0).permute(1, 0, 2, 3)     # [T,1,H,W]

        m = m.permute(1, 0, 2, 3)                # [1, T, H, W]

        if 反选:
            m = 1.0 - m

        # 扩展与羽化都在 latent 网格上做，半径按 16 折算（1 latent = 16 像素）
        r_grow = int(round(max(0, 扩展) / 16.0))
        r_feather = int(round(max(0, 羽化) / 16.0))
        if 扩展 < 0:
            # 负扩展 = 收缩，用最大池化做不了，改成先反选再膨胀再反选
            r_shrink = int(round(abs(扩展) / 16.0))
            if r_shrink > 0:
                m = 1.0 - _grow(1.0 - m, r_shrink)
        if r_grow > 0:
            m = _grow(m, r_grow)
        if r_feather > 0:
            m = _feather(m, r_feather)
        m = m.clamp(0.0, 1.0)

        # ---- 扩成和 video 潜空间同形 ----
        vmask = m.unsqueeze(0)                                   # [1,1,T,H,W]
        vmask = vmask.expand(B, C, T_lat, H_lat, W_lat).contiguous()

        # ---- audio 侧：沿用上游已有的，没有就全零 ----
        # 蒙版一律 float32：ComfyUI 的 noise_mask 按 float32 参与 1-mask 运算，
        # bf16 在 0.5 附近只有 8 位有效位，羽化边缘会抖成台阶。
        vmask = vmask.float()
        prev = av_latent.get("noise_mask")
        if prev is not None and (isinstance(prev, comfy.nested_tensor.NestedTensor)
                                 or getattr(prev, "is_nested", False)):
            pm = list(prev.unbind())
            pm[0] = vmask
            new_mask = comfy.nested_tensor.NestedTensor(tuple(pm))
            audio_note = "沿用上游音频蒙版"
        else:
            if len(members) > 1:
                audio_zero = torch.zeros_like(members[1], dtype=torch.float32)
                new_mask = comfy.nested_tensor.NestedTensor((vmask, audio_zero))
            else:
                new_mask = comfy.nested_tensor.NestedTensor((vmask,))
            audio_note = "音频侧写零（音轨保持）"

        out = dict(av_latent)
        out["noise_mask"] = new_mask

        cover = float(m.mean())
        report = (
            f"区域蒙版：像素 {t_px}x{h_px}x{w_px} -> latent {T_lat}x{H_lat}x{W_lat}\n"
            f"  扩展 {扩展}px / 羽化 {羽化}px / 反选 {反选}\n"
            f"  重绘面积占比 {cover*100:.1f}%，{audio_note}"
        )
        print("[区域蒙版] " + report)
        return (out, report)
