# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# MiniMax H3 本地图生视频：首帧/尾帧关键帧 + 外置音频参考。
# 首尾帧作为关键帧（几何锚定，minimax_keyframes）与 <Picture i> 视觉参考；
# 外置音频按官方 ref2va 的 ref_audio 条件注入（永远干净的参考音频行），
# 提示词里用 <Audio 1> 引用它，模型据此驱动口型并生成配套音频流。

import math

import torch
import torchaudio

import nodes
import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import node_helpers
from fxai_image_utils import normalize_images

FPS = 24
AUDIO_LATENT_FPS = 40
AUDIO_SAMPLE_RATE = 32000
CANVAS_MULTIPLE = 32
REF_IMAGE_SHORT_EDGE = 2048


def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length):
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


def _resize(image, width, height, crop):
    # image [B, H, W, C] -> [B, height, width, 3]
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _prepare_image(image, width, height, crop):
    # 已在外部对齐到目标尺寸的图原样使用（仍裁剪到 3 通道）；否则兜底缩放
    img = image[:1, ..., :3]
    if img.shape[1] != height or img.shape[2] != width:
        img = _resize(img, width, height, crop)
    return img


def _empty_av_latent(width, height, length, batch_size=1):
    frame_count, latent_t, audio_t = temporal_shape(length)
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16],
                        device=comfy.model_management.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


def _encode_ref_audio(audio_vae, audio):
    """外置音频 -> ([1, 32, 2, T] 音频潜变量, 参考时长 T)；mono 复制为立体声。"""
    waveform = audio["waveform"]  # [B, C, L]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", AUDIO_SAMPLE_RATE)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    waveform = waveform[:1]
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    z = audio_vae.encode(waveform.movedim(1, -1))  # [1, 32, 2, T]
    return z, z.shape[-1]


class FxAiMiniMaxImageToVideo:
    """MiniMax H3 图生视频：提示词 + 首尾帧关键帧 + 外置音频参考。

    输出正向条件与 AV 联合潜变量（视频[1,24,T,h,w] + 音频[1,32,2,T]）。
    接入外置音频时按 ref_audio 条件注入，提示词里写 <Audio 1> 引用；
    最终音频以源音频为准（外置音频走音频VAE 或用源文件 mux）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "CLIP模型": ("CLIP",),
                "视频VAE": ("VAE",),
                "提示词": ("STRING", {"forceInput": True}),
                "宽度": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "高度": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "帧数": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17,"tooltip": "24fps 帧数，自动对齐到模型的 17k+5 网格（124≈5秒；训练范围约124-362）"}),
            },
            "optional": {
                "音频VAE": ("VAE",),
                "首帧图片": ("IMAGE",),
                "尾帧图片": ("IMAGE",),
                "参考图片列表": ("IMAGE",),
                "外置音频": ("AUDIO",),
                "过渡帧列表": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("正向条件", "AV潜变量")
    FUNCTION = "run"
    CATEGORY = "凤希AI/视频"

    def run(self, CLIP模型, 视频VAE, 提示词, 宽度, 高度, 帧数,
            音频VAE=None, 首帧图片=None, 尾帧图片=None, 参考图片列表=None, 外置音频=None, 过渡帧列表=None):
        latent, frame_count = _empty_av_latent(宽度, 高度, 帧数)

        if 过渡帧列表 is not None:
            # 上一段落末帧整段编码后写入潜空间开头作为采样起点，音频部分不动（采样时音频仍走参考渲染）
            samples = latent["samples"]
            video, audio = samples.unbind()
            w, h = video.shape[4] * 16, video.shape[3] * 16
            init = 视频VAE.encode(_resize(过渡帧列表, w, h, "disabled"))
            k = min(init.shape[2], video.shape[2])
            new_video = video.clone()
            new_video[:, :, :k] = init[:, :, :k]
            latent["samples"] = comfy.nested_tensor.NestedTensor((new_video, audio))

        ref_items = []
        ref_blocks = []
        keyframes = []
        if 首帧图片 is not None:
            img = _prepare_image(首帧图片, 宽度, 高度, "disabled")
            keyframes.append({"resolved_frame_index": 0, "image": img})
        if 尾帧图片 is not None:
            img = _prepare_image(尾帧图片, 宽度, 高度, "center")
            keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})
        if 参考图片列表 is not None:
            for img in normalize_images(参考图片列表):
                h, w = img.shape[1], img.shape[2]
                scale = min(1.0, math.sqrt((宽度 * 高度) / (w * h)))
                tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                resized = _resize(img[:1], tw, th, "disabled")
                z = 视频VAE.encode(resized)
                ref_items.append({"type": "image", "data": resized})
                ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})
        if 外置音频 is not None:
            if 音频VAE is None:
                raise ValueError("接入外置音频时需连接音频VAE")
            audio_latent, ref_audio_t = _encode_ref_audio(音频VAE, 外置音频)
            ref_items.append({"type": "audio"})
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

        tokens = CLIP模型.tokenize(提示词, minimax_ref_items=ref_items)
        for i, item in enumerate(ref_items):
            if item.get("type") != "image":
                continue
            try:
                import comfy.text_encoders.qwen_vl as _qv
                fl, gr = _qv.process_qwen2vl_images(
                    item["data"], patch_size=16,
                    image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
            except Exception as e:
                print(f"[FxAiMiniMax] ref图#{i} 诊断失败: {e}")
        cond = CLIP模型.encode_from_tokens_scheduled(tokens)

        if keyframes:
            for kf in keyframes:
                kf["latent"] = 视频VAE.encode(kf.pop("image"))
            cond = node_helpers.conditioning_set_values(cond, {"minimax_keyframes": keyframes, "minimax_frame_count": frame_count})
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})

        return (cond, latent)
