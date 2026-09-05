# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# MiniMax H3 专用视频保存 V2：全部帧写入视频（不切过渡帧），
# 帧率固定 24、目录固定 sucai，默认取视频最后一帧作为过渡帧输出。

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

import fxai_task_store

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
        print(f"[凤希AI] FFmpeg音频转换失败：{str(e)}")
        import traceback
        traceback.print_exc()
        return ""

# 视频合成：全部帧写入，不切过渡帧
def save_video(images, save_dir, audio, fps=24, custom_num=0):
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
        print("[凤希AI] 视频路径安全校验失败，禁止写入")
        return ""

    img_np = (images.cpu().numpy() * 255).astype(np.uint8)
    total_frames = img_np.shape[0]

    if total_frames == 0:
        print("[凤希AI] 视频合成失败，没有有效帧")
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
    print(f"[凤希AI] 视频成功保存：{save_path}")
    return save_path


class FxAiMiniMaxVideoSaveV2:
    CATEGORY = "凤希AI/MiniMax"
    FUNCTION = "run"

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("过渡帧", "视频文件路径", "保存目录")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片序列": ("IMAGE",),
                "视频序号": ("INT", {"default": -1, "min": -1}),
                "音频": ("AUDIO",),
            },
        }

    def run(self, 图片序列, 视频序号, 音频):
        if 图片序列 is None:
            return (图片序列, "", "")

        target_dir = get_video_dir("sucai")

        # 全部帧进视频；过渡帧 = 整个图片序列的最后一帧（始终只返回一帧）
        video_images = 图片序列
        过渡帧 = 图片序列[-1:]

        video_path = save_video(
            images=video_images,
            save_dir=target_dir,
            audio=音频,
            fps=24,
            custom_num=视频序号,
        )

        del video_images, 图片序列
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if video_path and os.path.exists(video_path):
            try:
                if 视频序号 < 0:
                    # 保存结果（持久化+广播）
                    fxai_task_store.save_result("video", "sucai", [os.path.basename(video_path)])
                else:
                    # 广播过程信息：当前第几个场景
                    fxai_task_store.broadcast("scene_saved", {
                        "scene_index": 视频序号,
                        "scene_count": 视频序号 + 1,
                        "path": video_path,
                        "message": f"第 {视频序号 + 1} 个场景视频已生成"
                    })
            except Exception as e:
                print(f"[凤希AI] 视频广播失败：{e}")

        return (过渡帧, video_path, target_dir)
		
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")