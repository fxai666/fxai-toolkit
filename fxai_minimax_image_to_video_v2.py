# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# MiniMax H3 本地图生视频 V2：首帧/尾帧关键帧 + 外置音频参考。
# 首尾帧作为关键帧（几何锚定，minimax_keyframes）；
# 外置音频按官方 ref2va 的 ref_audio 条件注入（永远干净的参考音频行），
# 提示词里用 <Audio 1> 引用它，模型据此驱动口型并生成配套音频流。

import math
import re

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
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344


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


def _validate_media_tags(prompt, ref_items):
    """校验提示词里的 <Picture/Video/Audio N> 标签与已组装素材数量匹配。

    官方 tokenizer 按 ref_items 顺序独立计数（image->Picture、audio->Audio、video->Video）。
    提示词引用该类型的最大序号超出实际数量时，报错说明需要几个、实际传了几个。
    """
    counts = {"image": 0, "audio": 0, "video": 0}
    for item in ref_items:
        kind = item.get("type")
        if kind in counts:
            counts[kind] += 1
    tag_re = re.compile(r"<\s*(?:Picture|Image|Video|Audio)\s*(\d+)\s*>", re.IGNORECASE)
    needed = {"image": 0, "audio": 0, "video": 0}
    for m in tag_re.finditer(prompt or ""):
        media = re.sub(r"[^A-Za-z]", "", m.group(0)).lower()
        kind = "image" if media in ("picture", "image") else media
        if kind in needed:
            needed[kind] = max(needed[kind], int(m.group(1)))
    names = {"image": "图片(Picture)", "audio": "音频(Audio)", "video": "视频(Video)"}
    missing = []
    for kind, want in needed.items():
        if want > counts[kind]:
            missing.append(f"{names[kind]}需要 {want} 个，实际只有 {counts[kind]} 个")
    if missing:
        raise ValueError("提示词引用的素材不足：" + "；".join(missing))


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
    """外置音频 -> ([1, 32, 2, T] 音频潜变量, 参考时长 T)。

    DAC 编码器期望 stereo 双声道输入 [B, 2, L]；mono 先复制成双声道再编码。
    """
    waveform = audio["waveform"]  # [B, C, L]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", AUDIO_SAMPLE_RATE)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    if waveform.shape[-1] == 0:
        raise ValueError("参考音频为空（0 个采样），请检查接入的音频是否有效")
    z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [B, C, L] -> [B, L, C]，包装层 encode 期望
    return z, z.shape[-1]


def _fit_audio_latent(encoded, template):
    """把编码音频 latent 适配到目标 audio_t，与模板同 device/dtype。"""
    if encoded.shape[1:-1] != template.shape[1:-1]:
        raise ValueError(
            f"音频潜变量布局不匹配：got {tuple(encoded.shape)}，目标 {tuple(template.shape)}"
        )
    target_t = template.shape[-1]
    if encoded.shape[-1] > target_t:
        encoded = encoded[..., :target_t]
    elif encoded.shape[-1] < target_t:
        pad = encoded.new_zeros((*encoded.shape[:-1], target_t - encoded.shape[-1]))
        encoded = torch.cat((encoded, pad), dim=-1)
    return encoded.to(device=template.device, dtype=template.dtype)


def _lock_source_audio(latent, audio_vae, audio):
    """把源音频锁进 AV 潜空间：音频通道替换为源音频，noise_mask 音频部分全 0。"""
    samples = latent["samples"]
    video, template_audio = samples.unbind()
    encoded = _fit_audio_latent(_encode_ref_audio(audio_vae, audio)[0], template_audio)
    masks = latent.get("noise_mask")
    if masks is not None and getattr(masks, "is_nested", False):
        video_mask = tuple(masks.unbind())[0]
    elif isinstance(masks, torch.Tensor):
        video_mask = masks
    else:
        video_mask = torch.ones_like(video)
    audio_mask = torch.zeros_like(encoded)
    latent["samples"] = comfy.nested_tensor.NestedTensor((video, encoded))
    latent["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    return latent


class FxAiMiniMaxImageToVideoV2:
    """MiniMax H3 图生视频 V2：提示词 + 首尾帧关键帧 + 外置音频参考。

    输出正向条件与 AV 联合潜变量（视频[1,24,T,h,w] + 音频[1,32,2,T]）。
    接入外置音频时按 ref_audio 条件注入，提示词里写 <Audio 1> 引用；
    最终音频以源音频为准（外置音频走音频VAE 或用源文件 mux）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {"forceInput": True}),
                "CLIP模型": ("CLIP",),
                "视频VAE": ("VAE",),
                "宽度": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "高度": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "帧数": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17,"tooltip": "24fps 帧数，自动对齐到模型的 17k+5 网格（124≈5秒；训练范围约124-362）"}),
            },
            "optional": {
                "音频VAE": ("VAE",),
                "首帧图片": ("IMAGE",),
                "尾帧图片": ("IMAGE",),
                "外置音频": ("AUDIO",),
                "参考图片列表": ("IMAGE",),
                "参考音频列表": ("LIST",),
                "音频模式": ("COMBO", {
                    "options": ["音色参考", "原音频", "系统生成"],
                    "default": "音色参考",
                    "tooltip": "音色参考=外置音频仅作音色参考，模型生成内容；原音频=外置音频锁进音频通道，输出音频即源音频（唱歌/数字人口播）；系统生成=外置音频不参与，模型自由生成"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("正向条件", "AV潜变量")
    FUNCTION = "run"
    CATEGORY = "凤希AI/MiniMax"

    def run(self, CLIP模型, 视频VAE, 提示词, 宽度, 高度, 帧数,
            音频VAE=None, 首帧图片=None, 尾帧图片=None, 参考图片列表=None, 参考音频列表=None, 外置音频=None, 音频模式="音色参考"):
        latent, frame_count = _empty_av_latent(宽度, 高度, 帧数)

        if 音频模式 == "原音频":
            if 音频VAE is None:
                raise ValueError("原音频 模式需连接音频VAE")
            if 外置音频 is None:
                raise ValueError("原音频 模式需连接外置音频")
            latent = _lock_source_audio(latent, 音频VAE, 外置音频)
        elif 音频模式 == "音色参考" and 外置音频 is not None and 音频VAE is None:
            raise ValueError("音色参考 接入外置音频时需连接音频VAE")

        ref_items = []
        ref_blocks = []
        keyframes = []
        ref_images = []
        if 首帧图片 is not None:
            img = _prepare_image(首帧图片, 宽度, 高度, "disabled")
            keyframes.append({"resolved_frame_index": 0, "image": img})
            ref_images.append(img)
        if 尾帧图片 is not None:
            img = _prepare_image(尾帧图片, 宽度, 高度, "center")
            keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})
            ref_images.append(img)
        if 参考图片列表 is not None:
            ref_images += normalize_images(参考图片列表)

        for img in ref_images[:9]:
            h, w = img.shape[1], img.shape[2]
            scale = min(1.0, math.sqrt((宽度 * 高度) / (w * h)))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img[:1], tw, th, "disabled")
            z = 视频VAE.encode(resized)
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})

        # 参考音频：合并处理（外置音频放最前，再追加参考音频列表），逐个编码成独立 <Audio j> 参考，最多 3 段
        # 系统生成 模式下外置音频不参与；音色参考/原音频 才作为参考条件
        ref_audios = []
        if 外置音频 is not None and 音频模式 != "系统生成" and 外置音频.get("waveform", None) is not None and 外置音频["waveform"].shape[-1] > 0:
            ref_audios.append(外置音频)
        for audio in (参考音频列表 or []):
            if isinstance(audio, dict) and "waveform" in audio and audio["waveform"].shape[-1] > 0:
                ref_audios.append(audio)
        if ref_audios:
            if 音频VAE is None:
                raise ValueError("接入参考音频时需连接音频VAE")
            _, template_audio = latent["samples"].unbind()
            for i, audio in enumerate(ref_audios[:3]):
                audio_latent, ref_audio_t = _encode_ref_audio(音频VAE, audio)
                # 外置音频（首个）fit 到目标音频时长，保证 ref_audio_t 与目标 audio_t
                # 对齐（GH drive_audio 同款处理）；参考音频列表保持原长度
                if i == 0 and 外置音频 is not None and 音频模式 != "系统生成":
                    audio_latent = _fit_audio_latent(audio_latent, template_audio)
                    ref_audio_t = int(audio_latent.shape[-1])
                ref_items.append({"type": "audio"})
                ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

        _validate_media_tags(提示词, ref_items)
        tokens = CLIP模型.tokenize(提示词, minimax_ref_items=ref_items)
        cond = CLIP模型.encode_from_tokens_scheduled(tokens)

        if keyframes:
            for kf in keyframes:
                if "image" in kf:
                    kf["latent"] = 视频VAE.encode(kf.pop("image"))
            cond = node_helpers.conditioning_set_values(cond, {"minimax_keyframes": keyframes, "minimax_frame_count": frame_count})
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})

        return (cond, latent)