# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# MiniMax H3 Reference to Video：图片列表 + 单个音频 -> 条件 + AV 潜变量。
# 多张参考图按 minimax_refs 以 image 块注入（提示词 <Picture 1>/<Picture 2>...），
# 音频按官方 ref2va 的 ref_audio 条件注入（提示词 <Audio 1>），
# 用于逐段循环生成长视频：每段把上一段末帧追加进图片列表作参考。

import math

import torch
import torchaudio

import nodes
import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import node_helpers

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
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _empty_av_latent(width, height, length, batch_size=1):
    frame_count, latent_t, audio_t = temporal_shape(length)
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16],
                        device=comfy.model_management.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


def _encode_ref_audio(audio_vae, audio):
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


class FxAiMiniMaxH3RefToVideo:
    """MiniMax H3 Reference to Video：图片列表 + 单个音频。

    图片列表按 <Picture i> 顺序作为参考图注入，音频作为 <Audio 1> 参考注入；
    输出正向条件与 AV 联合潜变量。长视频逐段循环生成时，
    把上一段的末帧追加进图片列表即可延续画面。
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
                "帧数": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17, "tooltip": "24fps 帧数，自动对齐到模型的 17k+5 网格（124≈5秒；训练范围约124-362）"}),
            },
            "optional": {
                "音频VAE": ("VAE",),
                "参考图片列表": ("IMAGE",),
                "参考音频": ("AUDIO",),
                "参考图片尺寸": (["match", "max"],),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("正向条件", "AV潜变量")
    FUNCTION = "run"
    CATEGORY = "凤希AI/视频"

    def run(self, CLIP模型, 视频VAE, 提示词, 宽度, 高度, 帧数,
            音频VAE=None, 参考图片列表=None, 参考音频=None, 参考图片尺寸="match"):
        latent, _frame_count = _empty_av_latent(宽度, 高度, 帧数)

        ref_items = []
        ref_blocks = []
        if 参考图片列表 is not None:
            for img in 参考图片列表:
                img = img[None, ...]
                h, w = img.shape[1], img.shape[2]
                if 参考图片尺寸 == "match":
                    scale = min(1.0, math.sqrt((宽度 * 高度) / (w * h)))
                else:
                    scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
                tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                resized = _resize(img[:1], tw, th, "disabled")
                z = 视频VAE.encode(resized)
                ref_items.append({"type": "image", "data": resized})
                ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})

        if 参考音频 is not None:
            if 音频VAE is None:
                raise ValueError("接入参考音频时需连接音频VAE")
            audio_latent, ref_audio_t = _encode_ref_audio(音频VAE, 参考音频)
            ref_items.append({"type": "audio"})
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

        tokens = CLIP模型.tokenize(提示词, minimax_ref_items=ref_items)
        cond = CLIP模型.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
        return (cond, latent)
