# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# MiniMax H3 专用视频保存：全部帧写入视频（不切过渡帧），
# 默认取视频最后一帧作为过渡帧输出给下一段做首帧/参考图。

import os
import re
import torch
import numpy as np
from PIL import Image
import folder_paths
import subprocess
import tempfile
import io
import gc
from datetime import datetime

# 安全路径校验
def safe_path_join(base_dir, path):
    base_dir = os.path.abspath(base_dir)
    full_path = os.path.abspath(os.path.join(base_dir, path))
    if not full_path.startswith(base_dir):
        return None
    return full_path

# 获取视频保存目录
def get_video_dir(subdir=""):
    comfy_root = folder_paths.base_path
    base_dir = "fxai/video"
    target_dir = os.path.join(comfy_root, base_dir)

    if subdir:
        subdir = re.sub(r'[\\/*?:"<>|]', "", subdir)
        target_dir = os.path.join(target_dir, subdir)

    os.makedirs(target_dir, exist_ok=True)
    return target_dir

# 获取全局临时音频路径
def get_fixed_temp_audio_path():
    comfy_root = folder_paths.base_path
    temp_dir = os.path.join(comfy_root, "fxai/video/temp")
    os.makedirs(temp_dir, exist_ok=True)
    return os.path.join(temp_dir, "fxai_temp_audio.wav")

# 音频去噪档位（FFmpeg afftdn 参数）
AUDIO_DENOISE_LEVELS = {
    "关闭": None,
    "轻度": {"nr": 6, "nf": -50},
    "标准": {"nr": 12, "nf": -50},
    "强力": {"nr": 24, "nf": -50},
}

# 用 FFmpeg afftdn 抑制稳态底噪/嘶声，逐声道独立处理
def denoise_audio_ffmpeg(waveform, sr, nr, nf):
    b, c, l = waveform.shape
    wav = waveform.detach().float().cpu().numpy()
    data = np.ascontiguousarray(wav.reshape(b * c, l).T).tobytes()
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'f32le', '-ar', str(int(sr)), '-ac', str(b * c), '-i', '-',
        '-af', f'afftdn=nr={nr}:nf={nf}',
        '-f', 'f32le', '-ar', str(int(sr)), '-ac', str(b * c), '-',
    ]
    proc = subprocess.run(cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    out = np.frombuffer(proc.stdout, dtype=np.float32).reshape(l, b * c).T.reshape(b, c, l)
    return torch.from_numpy(out).to(device=waveform.device, dtype=waveform.dtype)

# 音频张量转WAV
def audio_tensor_to_wav_ffmpeg(audio_dict):
    try:
        waveform = audio_dict["waveform"]
        sample_rate = audio_dict["sample_rate"]

        if waveform.ndim == 3 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)

        waveform_np = waveform.cpu().numpy()

        if waveform_np.ndim == 1:
            channels = 1
            audio_data = waveform_np.astype(np.float32)
        else:
            channels = waveform_np.shape[0]
            audio_data = np.ascontiguousarray(waveform_np.T).astype(np.float32)

        raw_pcm = audio_data.tobytes()
        temp_path = get_fixed_temp_audio_path()

        cmd = [
            'ffmpeg', '-y',
            '-f', 'f32le',
            '-ar', str(sample_rate),
            '-ac', str(channels),
            '-i', 'pipe:0',
            '-c:a', 'pcm_s16le',
            temp_path
        ]

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        try:
            proc.stdin.write(raw_pcm)
        finally:
            proc.stdin.close()
            proc.wait()

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

        return temp_path
    except Exception as e:
        print(f"[凤希AI FFmpeg音频转换失败] {str(e)}")
        import traceback
        traceback.print_exc()
        return ""

# 视频合成：全部帧写入，不切过渡帧
def save_video(images, save_dir, audio, fps=24, custom_num=0, denoise="关闭"):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if custom_num < 0:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fxai_{time_str}.mp4"
    else:
        filename = f"{custom_num:03d}.mp4"

    save_path = safe_path_join(save_dir, filename)
    if save_path is None:
        print("[凤希AI视频] 路径安全校验失败，禁止写入")
        return ""

    img_np = (images.cpu().numpy() * 255).astype(np.uint8)
    total_frames = img_np.shape[0]

    if total_frames == 0:
        print("[凤希AI视频合成失败] 没有有效帧")
        return ""

    try:
        height, width = img_np[0].shape[0], img_np[0].shape[1]
        video_duration = total_frames / fps

        if isinstance(audio, dict) and "waveform" in audio:
            waveform = audio["waveform"]
            sample_rate = audio.get("sample_rate", 0)
            max_sample_count = int(video_duration * sample_rate) if sample_rate > 0 else 0

            if max_sample_count > 0 and waveform.size(-1) > max_sample_count:
                waveform = waveform[..., :max_sample_count]

            level = AUDIO_DENOISE_LEVELS.get(denoise)
            if level is not None:
                waveform = denoise_audio_ffmpeg(waveform, sample_rate, **level)

            audio = {"waveform": waveform, "sample_rate": sample_rate}
            audio = audio_tensor_to_wav_ffmpeg(audio)

        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', '-',
        ]
        if isinstance(audio, str) and os.path.exists(audio):
            cmd += ['-i', audio, '-c:a', 'aac', '-b:a', '192k']
        cmd += [
            '-c:v', 'libx264',
            '-preset', 'slow',
            '-crf', '17',
            '-pix_fmt', 'yuv420p',
            '-frames:v', str(total_frames),
            '-movflags', '+faststart',
            save_path
        ]

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=1024*1024*10
        )

        try:
            batch_size = 20
            for i in range(0, len(img_np), batch_size):
                batch = img_np[i:i+batch_size]
                batch_data = b''.join([img.tobytes() for img in batch])
                proc.stdin.write(batch_data)
        finally:
            proc.stdin.close()
            proc.wait()

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

    except Exception as e:
        print(f"[凤希AI视频合成失败] {str(e)}")
        import traceback
        traceback.print_exc()
        return ""

    gc.collect()
    print(f"[凤希AI视频] 成功保存：{save_path}")
    return save_path


class FxAiMiniMaxVideoSave:
    CATEGORY = "凤希AI/MiniMax"
    FUNCTION = "run"

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "INT", "AUDIO")
    RETURN_NAMES = ("过渡帧", "过渡帧列表", "视频文件路径", "保存目录", "实际帧数", "过渡帧音频")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片序列": ("IMAGE",),
                "目录": ("STRING", {"default": "sucai"}),
                "生成帧数": ("INT", {"default": 227, "min": 6, "max": 3600, "step": 17,
                    "tooltip": "进视频的帧数（17k+5 网格）。视频取前 N 帧；过渡帧列表固定取图片序列最后 24 帧，过渡帧取整个图片序列的最后一帧。"}),
                "帧率FPS": ("INT", {"default": 24, "min": 1}),
                "视频序号": ("INT", {"default": -1, "min": -1}),
                "音频": ("AUDIO",),
                "音频去噪": ("COMBO", {
                    "options": list(AUDIO_DENOISE_LEVELS),
                    "default": "关闭",
                    "tooltip": "对输出音频用 FFmpeg afftdn 抑制稳态底噪/嘶声"}),
            },
        }

    def run(self, 图片序列, 目录, 生成帧数, 帧率FPS, 视频序号, 音频, 音频去噪="关闭"):
        if 图片序列 is None:
            return (图片序列, 图片序列, "", "", 0, None)

        target_dir = get_video_dir(目录)
        total_frames = len(图片序列)
        保存帧数 = min(int(生成帧数), max(0,total_frames))

        # 视频 = 前 生成帧数 帧（不足则全部）
        video_images = 图片序列[:保存帧数]
        # 过渡帧列表 = 图片序列的最后 22 帧
        过渡帧列表 = 图片序列[-22:]

        # 过渡帧 = 整个图片序列的最后一帧（始终只返回一帧）
        过渡帧 = 图片序列[-1:]

        video_path = save_video(
            images=video_images,
            save_dir=target_dir,
            audio=音频,
            fps=帧率FPS,
            custom_num=视频序号,
            denoise=音频去噪,
        )

        # 过渡帧音频 = 与过渡帧列表同长的末尾音频（作为背景音延续参考）
        tail_audio = None
        if isinstance(音频, dict) and "waveform" in 音频:
            wf = 音频["waveform"]
            sr = 音频.get("sample_rate", 0)
            n = int(len(过渡帧列表) / 帧率FPS * sr) if sr > 0 else wf.shape[-1]
            n = min(wf.shape[-1], n)
            tail_audio = {"waveform": wf[..., -n:], "sample_rate": sr}

        del video_images, 图片序列
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (过渡帧, 过渡帧列表, video_path, target_dir, 保存帧数, tail_audio)