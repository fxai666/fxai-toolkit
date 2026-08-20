# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# MiniMax H3 本地图生视频：首帧/尾帧关键帧 + 外置音频参考。
# 首尾帧作为关键帧（几何锚定，minimax_keyframes）与 <Picture i> 视觉参考；
# 外置音频按官方 ref2va 的 ref_audio 条件注入（永远干净的参考音频行），
# 提示词里用 <Audio 1> 引用它，模型据此驱动口型并生成配套音频流。

import math
import re
import subprocess

import numpy as np
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
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344


def adapt_canvas(width, height):
    """768-short-edge canvas with 768*1344 area cap, per-axis round to 32."""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


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


def _split_ref_videos(value):
    """参考视频列表 -> 多个视频帧序列 [T,H,W,C] 的列表。

    兼容两种输入形态：
    - 纯 IMAGE 批 [T,H,W,C]：整批视为一个视频的帧序列
    - list[images, images]：每个元素一段视频帧序列，逐个分离
    """
    if isinstance(value, torch.Tensor):
        return [value]
    videos = []
    for img in value:
        if not isinstance(img, torch.Tensor):
            continue
        videos.append(img[:1] if img.dim() == 3 else img)
    return videos


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


AUDIO_DENOISE_LEVELS = {
    "关闭": None,
    "轻度": {"nr": 6, "nf": -50},
    "标准": {"nr": 12, "nf": -50},
    "强力": {"nr": 24, "nf": -50},
}


def _denoise_audio_ffmpeg(waveform, sr, nr, nf):
    """用 FFmpeg afftdn 抑制稳态底噪/嘶声，逐声道独立处理。"""
    b, c, l = waveform.shape
    wav = waveform.detach().float().cpu().numpy()
    data = np.ascontiguousarray(wav.reshape(b * c, l).T).tobytes()
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "f32le", "-ar", str(int(sr)), "-ac", str(b * c), "-i", "-",
        "-af", f"afftdn=nr={nr}:nf={nf}",
        "-f", "f32le", "-ar", str(int(sr)), "-ac", str(b * c), "-",
    ]
    proc = subprocess.run(cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    out = np.frombuffer(proc.stdout, dtype=np.float32).reshape(l, b * c).T.reshape(b, c, l)
    return torch.from_numpy(out).to(device=waveform.device, dtype=waveform.dtype)


def _encode_ref_audio(audio_vae, audio, denoise="关闭"):
    """外置音频 -> ([1, 32, 2, T] 音频潜变量, 参考时长 T)。

    DAC 编码器期望 stereo 双声道输入 [B, 2, L]；mono 先复制成双声道再编码。
    denoise 为可选 FFmpeg afftdn 去噪档位（AUDIO_DENOISE_LEVELS 键）。
    """
    waveform = audio["waveform"]  # [B, C, L]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", AUDIO_SAMPLE_RATE)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    level = AUDIO_DENOISE_LEVELS.get(denoise)
    if level is not None:
        waveform = _denoise_audio_ffmpeg(waveform, vae_sr, **level)
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
                "参考视频列表": ("IMAGE",),
                "参考视频音频": ("AUDIO",),
                "外置音频": ("AUDIO",),
                "参考音频列表": ("LIST",),
                "过渡帧列表": ("IMAGE",),
                "过渡羽化": ("INT", {"default": -1, "min": -1, "max": 5, "step": 1,
                    "tooltip": "过渡帧锁死区到自由区的平滑宽度（latent 步，约每步4帧）。-1=自动收紧（锁死前2步、只放宽1-2步，暗带最短）；0=硬锁；1-5=固定羽化步数，超过过渡帧折算步数无意义。"}),
                "音频模式": ("COMBO", {
                    "options": ["音色参考", "原音频", "系统生成"],
                    "default": "音色参考",
                    "tooltip": "音色参考=外置音频仅作音色参考，模型生成内容；原音频=外置音频锁进音频通道，输出音频即源音频（唱歌/数字人口播）；系统生成=外置音频不参与，模型自由生成"}),
                "音频去噪": ("COMBO", {
                    "options": list(AUDIO_DENOISE_LEVELS),
                    "default": "关闭",
                    "tooltip": "对参考音频用 FFmpeg afftdn 抑制稳态底噪/嘶声（影响音色参考质量，不影响原音频输出本身）"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("正向条件", "AV潜变量")
    FUNCTION = "run"
    CATEGORY = "凤希AI/MiniMax"

    def run(self, CLIP模型, 视频VAE, 提示词, 宽度, 高度, 帧数,
            音频VAE=None, 首帧图片=None, 尾帧图片=None, 参考图片列表=None, 参考视频列表=None, 参考视频音频=None, 外置音频=None, 参考音频列表=None, 过渡帧列表=None, 过渡羽化=-1, 音频模式="参考音色", 音频去噪="关闭"):
        latent, frame_count = _empty_av_latent(宽度, 高度, 帧数)

        if 过渡帧列表 is not None and 过渡帧列表.shape[0] > 0:
            # 将过渡帧编码写入潜空间开头并软锁：t=0 起前 k 步的 denoise_mask 从 0 平滑
            # 升到 1（前 lock_steps 步强锁、之后 feather 步渐放），消除硬边界产生的闪光，
            # 使锁死区画面平滑过渡到后续自由演化。羽化步数 -1 时自动收紧：锁死前 2 步、
            # 仅放宽 1-2 步，使暗带（两部分画面按细分权重混合的叠影）压到最短；
            # 显式值则按该步数羽化。
            samples = latent["samples"]
            video, audio = samples.unbind()
            w, h = video.shape[4] * 16, video.shape[3] * 16
            init = 视频VAE.encode(_resize(过渡帧列表, w, h, "disabled"))
            k = min(init.shape[2], video.shape[2])
            new_video = video.clone()
            new_video[:, :, :k] = init[:, :, :k][:, :, :k]
            latent["samples"] = comfy.nested_tensor.NestedTensor((new_video, audio))
            mask = torch.ones([1, 1, video.shape[2], video.shape[3], video.shape[4]],
                              dtype=torch.float32, device=video.device)
            feather = k if 过渡羽化 < 0 else min(过渡羽化, k)
            if 过渡羽化 < 0:
                lock_steps = min(2, k)
                feather = min(2, max(0, k - lock_steps))
            else:
                lock_steps = max(0, k - feather)
            if lock_steps > 0:
                mask[:, :, :lock_steps, :, :] = 0.0
            for i in range(feather):
                mask[:, :, lock_steps + i, :, :] = (i + 1) / (feather + 1)
            latent["noise_mask"] = mask

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
        if 首帧图片 is not None:
            img = _prepare_image(首帧图片, 宽度, 高度, "disabled")
            keyframes.append({"resolved_frame_index": 0, "image": img})
        if 尾帧图片 is not None:
            img = _prepare_image(尾帧图片, 宽度, 高度, "center")
            keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})
        ref_images = []
        if 参考图片列表 is not None:
            ref_images += normalize_images(参考图片列表)[:9]
        if 过渡帧列表 is not None and 过渡帧列表.shape[0] > 0:
            ref_images += normalize_images(过渡帧列表)[:9]
        for img in ref_images:
            h, w = img.shape[1], img.shape[2]
            scale = min(1.0, math.sqrt((宽度 * 高度) / (w * h)))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img[:1], tw, th, "disabled")
            z = 视频VAE.encode(resized)
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})
        video_audio_blocks = []
        if 参考视频列表 is not None:
            for idx, video_frames in enumerate(_split_ref_videos(参考视频列表)):
                if video_frames.shape[0] > frame_count:
                    video_frames = video_frames[:frame_count]
                n = video_frames.shape[0]
                if n < 5:
                    video_frames = video_frames[-1:].repeat(5, 1, 1, 1)
                    n = 5
                while n % 17 != 5:
                    n -= 1
                video_frames = video_frames[:n]
                vh, vw = video_frames.shape[1], video_frames.shape[2]
                cw, ch = adapt_canvas(vw, vh)
                if vw * vh < cw * ch:
                    cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                    ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                resized = _resize(video_frames, cw, ch, "disabled")
                z = 视频VAE.encode(resized)
                sample_idx = list(range(0, resized.shape[0], FPS // 2))
                qwen_frames = resized[sample_idx]
                ref_items.append({"type": "video", "data": qwen_frames,
                                  "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
                # 参考视频可配对一段音频（官方 video_audio 块：音频行与视频共享时间原点）。
                # 块延迟到参考音频之后入列，使提示词 <Audio N> 仍对应参考音频列表
                if idx == 0 and isinstance(参考视频音频, dict) and "waveform" in 参考视频音频:
                    if 音频VAE is None:
                        raise ValueError("参考视频接入音频时需连接音频VAE")
                    encoded_soundtrack, soundtrack_t = _encode_ref_audio(音频VAE, 参考视频音频, denoise=音频去噪)
                    video_audio_blocks.append({"kind": "video_audio",
                                               "latent_t": z.shape[2],
                                               "latent_h": ch // 16, "latent_w": cw // 16,
                                               "ref_audio_t": soundtrack_t, "latent": z,
                                               "audio_latent": encoded_soundtrack})
                else:
                    ref_blocks.append({"kind": "video", "latent_t": z.shape[2],
                                       "latent_h": ch // 16, "latent_w": cw // 16,
                                       "ref_audio_t": 0, "latent": z, "audio_latent": None})
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
                audio_latent, ref_audio_t = _encode_ref_audio(音频VAE, audio, denoise=音频去噪)
                # 外置音频（首个）fit 到目标音频时长，保证 ref_audio_t 与目标 audio_t
                # 对齐（GH drive_audio 同款处理）；参考音频列表保持原长度
                if i == 0 and 外置音频 is not None and 音频模式 != "系统生成":
                    audio_latent = _fit_audio_latent(audio_latent, template_audio)
                    ref_audio_t = int(audio_latent.shape[-1])
                ref_items.append({"type": "audio"})
                ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})
        # 视频音频块最后入列：参考音频列表保持 <Audio 1..N>，视频音频接在末尾
        for blk in video_audio_blocks:
            ref_items.append({"type": "audio"})
            ref_blocks.append(blk)

        _validate_media_tags(提示词, ref_items)
        tokens = CLIP模型.tokenize(提示词, minimax_ref_items=ref_items)
        cond = CLIP模型.encode_from_tokens_scheduled(tokens)

        if keyframes:
            for kf in keyframes:
                if "image" in kf:
                    kf["latent"] = 视频VAE.encode(kf.pop("image"))
            cond = node_helpers.conditioning_set_values(
                cond, {"minimax_keyframes": keyframes, "minimax_frame_count": frame_count})
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})

        return (cond, latent)
