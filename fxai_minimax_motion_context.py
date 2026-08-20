# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# MiniMax上下文衔接（FxAiMiniMaxMotionContext）：镜头链上下文节点（思路借鉴 ComfyUI-H3-Motion-Context）。
#
# 每个镜头生成"目标帧 + 锚定帧"总长的 AV latent（例如 73+17=90 帧），本节点把生成的完整 latent
# 直接在 latent 层面（含视频+音频双流，不解码/不重编码/不 resize）切成两段：
#   - 成品段（成品 latent）  = 前 目标帧 → 解码为成片视频
#   - 锚定段（锚定 latent）  = 尾 锚定帧 → 存盘，作为下一镜头部的锚定源（交给图生视频节点）
#
# 图生视频节点（fxai_minimax_image_to_video）接收上一镜的锚定段 latent，作为本镜头部内部 keyframe
# 锚点（pinned，锁死不参与去噪）+ 音频延续，模型从上一镜画面真正继续。全程 latent 层面，无损，
# 从根上消除逐镜像素级有损重建造成的劣化/锐化。拼接 = 镜A成品段 + 镜B成品段，无缝无重复。
#
# 提示词：衔接相关措辞（Picture 4 / keyframe completion / begins from / Video 1 / Audio 3）全部删除，
# 提示词只描述本镜内容；P1/P2/P3 人物/场景参考图 + Audio 1/2 音色参考仍保留。

import torch

from fxai_minimax_core_patch import (
    _steps_for_frames,
)

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FPS = 24
ANCHOR_FRAMES = 17  # 锚定帧数：17 帧 = 5 latent 步，与 17k+5 网格精确对齐（完整周期）


class FxAiMiniMaxMotionContext:
    """从本镜生成完的 AV latent 直接切片，切出【成品段 + 锚定段】。

    输入本镜 SamplerCustomAdvanced 的输出（总帧 = 目标帧 + 锚定帧），直接在 latent 层面切分，
    视频 + 音频双流同步切，不解码不重编码：
      - 成品段：前 目标帧 → 解码成片
      - 锚定段：尾 锚定帧 latent → 存盘给下一镜（图生视频节点）当锚定源
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("实际潜空间", "引导潜空间")
    FUNCTION = "split"
    CATEGORY = "凤希AI/MiniMax"
    DESCRIPTION = ("把本镜生成完的 AV latent（视频+音频双流）直接在 latent 层面切成："
                   "前段成品 latent（解码成片）+ 尾部 17 帧锚定 latent（存盘给下一镜当锚定源）。")

    def split(self, latent):
        video = self._video_from_latent(latent)
        latent_t = int(video.shape[2])
        total_frames = self._pixel_frames(latent_t)
        n = ANCHOR_FRAMES

        # 锚定段必须是整数 latent 步；成品段 = 总步 - 锚定步
        anchor_steps = _steps_for_frames(n)
        if anchor_steps is None:
            raise ValueError("MiniMax上下文衔接: 锚定 %d 帧不是整数 latent 步（可用 5/22/39/56）" % n)
        if anchor_steps >= latent_t:
            raise ValueError("MiniMax上下文衔接: 锚定 %d 步 >= 总 %d 步" % (anchor_steps, latent_t))
        prod_steps = latent_t - anchor_steps
        prod_frames = self._pixel_frames(prod_steps)

        # 直接在 latent 层面切锚定段（尾 n 帧）与成品段（前 prod 步）
        parts = self._streams_from_latent(latent)
        video = parts[0]
        if video.ndim == 4:
            video = video.unsqueeze(0)
        anchor_video = video[:1, :, -anchor_steps:].clone()
        prod_video = video[:1, :, :prod_steps].clone()

        anchor_latent = {"samples": None}
        prod_latent = {"samples": None}
        if len(parts) >= 2:
            audio = parts[1]
            if audio.ndim == 3:
                audio = audio.unsqueeze(0)
            # 音频按视频帧比例同步切：锚定段音频 = 尾 n 帧对应，成品段 = 前 prod 帧对应
            total_audio_t = int(audio.shape[-1])
            anchor_audio_t = int(round(total_audio_t * n / total_frames))
            anchor_audio = audio[:1, ..., -anchor_audio_t:].clone() if anchor_audio_t > 0 else audio[:1, ..., :0].clone()
            prod_audio = audio[:1, ..., :total_audio_t - anchor_audio_t].clone()
            anchor_latent = {"samples": self._nest(anchor_video, anchor_audio)}
            prod_latent = {"samples": self._nest(prod_video, prod_audio)}
        else:
            anchor_latent = {"samples": anchor_video}
            prod_latent = {"samples": prod_video}

        return (prod_latent, anchor_latent)

    def _nest(self, video, audio=None):
        if audio is None:
            return video
        try:
            import comfy.nested_tensor
            return comfy.nested_tensor.NestedTensor((video, audio))
        except Exception:
            return (video, audio)

    def _streams_from_latent(self, latent):
        samples = latent["samples"]
        if hasattr(samples, "unbind"):
            return list(samples.unbind())
        if isinstance(samples, (tuple, list)):
            return list(samples)
        return [samples]

    def _video_from_latent(self, latent):
        samples = latent["samples"]
        if hasattr(samples, "unbind"):
            parts = list(samples.unbind())
        elif isinstance(samples, (tuple, list)):
            parts = list(samples)
        else:
            raise ValueError("MiniMax上下文衔接: 期望 H3 AV latent，实际 %r" % type(samples))
        video = parts[0]
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError("MiniMax上下文衔接: 期望视频 latent [B,C,T,H,W]，实际 %s"
                             % (tuple(video.shape),))
        return video

    def _pixel_frames(self, latent_t):
        return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


class FxAiMiniMaxMotionContextTrim:
    """裁掉头部锚定帧，画面与音频同步，防止音画错位累积。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "trim_frames": ("INT", {"default": 0, "min": 0, "max": 4096}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001}),
                "match_tail": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "trim"
    CATEGORY = "凤希AI/MiniMax"

    def trim(self, images, trim_frames, audio=None, fps=24.0, match_tail=True):
        n = max(0, int(trim_frames))
        total = int(images.shape[0])
        if n >= total:
            raise ValueError("MiniMax上下文衔接: 要裁 %d 帧，只剩 %d 帧" % (n, total))
        out_images = images[n:] if n else images
        out_audio = audio
        if audio is not None:
            waveform = audio["waveform"]
            sr = int(audio["sample_rate"])
            cut = int(round(n / float(fps) * sr))
            if cut >= int(waveform.shape[-1]):
                raise ValueError("MiniMax上下文衔接: 音频比裁切窗口还短")
            waveform = waveform[..., cut:]
            if match_tail:
                frames_left = total - n
                want = int(round(frames_left / float(fps) * sr))
                have = int(waveform.shape[-1])
                if have > want:
                    waveform = waveform[..., :want]
                elif have < want:
                    waveform = torch.nn.functional.pad(waveform, (0, want - have))
            out_audio = {"waveform": waveform, "sample_rate": sr}
        return (out_images, out_audio)


NODE_CLASS_MAPPINGS = {
    "FxAiMiniMaxMotionContext": FxAiMiniMaxMotionContext,
    "FxAiMiniMaxMotionContextTrim": FxAiMiniMaxMotionContextTrim,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FxAiMiniMaxMotionContext": "凤希AI - MiniMax上下文衔接",
    "FxAiMiniMaxMotionContextTrim": "凤希AI - MiniMax镜头链裁剪",
}