import os
import re
import time
import shutil
import subprocess
import torch
import numpy as np
import folder_paths
import gc

def safe_path_join(base_dir, path):
    base_dir = os.path.abspath(base_dir)
    full_path = os.path.abspath(os.path.join(base_dir, path))
    return full_path if full_path.startswith(base_dir) else None

def get_merge_output_dir():
    comfy_root = folder_paths.base_path
    target_dir = os.path.join(comfy_root, "fxai/video/merged")
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

def get_fixed_temp_audio_path():
    comfy_root = folder_paths.base_path
    temp_dir = os.path.join(comfy_root, "fxai/video/temp")
    os.makedirs(temp_dir, exist_ok=True)
    return os.path.join(temp_dir, "fxai_merge_temp_audio.wav")

# ==========================
# ✅ 修复版：不用管道 pipe:0，彻底解决 Windows 崩溃
# ==========================
def audio_tensor_to_wav_ffmpeg(audio_dict):
    gc.collect()
    try:
        waveform = audio_dict["waveform"]
        sample_rate = audio_dict["sample_rate"]

        if waveform.ndim == 3 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)

        waveform_np = waveform.cpu().numpy().astype(np.float32)

        if waveform_np.ndim == 1:
            audio_data = np.stack([waveform_np, waveform_np], axis=1)
        else:
            channels, samples = waveform_np.shape
            if channels == 1:
                mono = waveform_np[0]
                audio_data = np.stack([mono, mono], axis=1)
            else:
                audio_data = waveform_np[:2].T

        audio_data = np.ascontiguousarray(audio_data)
        temp_path = get_fixed_temp_audio_path()

        # 先写临时PCM文件 → 喂给ffmpeg（Windows 100%稳定）
        pcm_temp = temp_path + ".pcm"
        with open(pcm_temp, "wb") as f:
            f.write(audio_data.tobytes())

        cmd = [
            'ffmpeg', '-y',
            '-hide_banner', '-loglevel', 'error',
            '-f', 'f32le',
            '-ar', str(sample_rate),
            '-ac', '2',
            '-i', pcm_temp,
            '-c:a', 'pcm_s16le',
            temp_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        os.remove(pcm_temp)

        if result.returncode == 0 and os.path.exists(temp_path):
            return temp_path
        else:
            print(f"[凤希AI音频] ffmpeg错误: {result.stderr}")
            return ""

    except Exception as e:
        print(f"[凤希AI音频转换失败] {e}")
        return ""

# --------------------------
# ✅ 核心改动1：对齐音频替换的编码参数
# --------------------------
def replace_video_audio(video_path, audio_path):
    if not os.path.exists(video_path) or not os.path.exists(audio_path):
        return video_path

    temp_video = video_path.replace(".mp4", "_temp.mp4")
    if os.path.exists(temp_video):
        os.remove(temp_video)
        
    try:
        # 对齐视频生成模块的音频编码参数：aac + 192k码率 + movflags faststart
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', video_path, '-i', audio_path,
            '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',  # 对齐视频编码参数
            '-pix_fmt', 'yuv420p',  # 对齐像素格式
            '-c:a', 'aac', '-b:a', '192k', '-ac', '2',  # 对齐音频参数
            '-map', '0:v:0', '-map', '1:a:0', '-shortest',
            '-movflags', '+faststart',  # 对齐快速启动参数
            temp_video
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        shutil.move(temp_video, video_path)
    except Exception as e:
        print(f"[凤希AI音频替换失败] {e}")
        if os.path.exists(temp_video):
            os.remove(temp_video)
    return video_path

def get_video_files(source_dir, max_count=0):
    if not os.path.isdir(source_dir):
        return []
    
    exts = ('.mp4', '.webm', '.mov', '.avi')
    files = sorted(f for f in os.listdir(source_dir) if f.lower().endswith(exts))
    if max_count > 0:
        files = files[:max_count]
    return [safe_path_join(source_dir, f) for f in files]

# --------------------------
# ✅ 核心改动2：对齐视频合并的编码参数
# --------------------------
def merge_videos(source_dir, output_name, max_count=0, audio=None):
    gc.collect()
    videos = []
    list_path = None
    output_path = None
    audio_wav = None
    temp_concat = None

    try:
        videos = get_video_files(source_dir, max_count)
        output_dir = get_merge_output_dir()
        output_name = re.sub(r'[\\/*?:"<>|]', "", output_name.strip())
        output_path = safe_path_join(output_dir, f"{output_name}.mp4")

        if not videos:
            print("[凤希AI视频合并] 无视频文件")
            return None

        # 生成拼接列表
        list_path = os.path.join(source_dir, "merge_list.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for p in videos:
                f.write(f"file '{p.replace('\\', '/')}'\n")

        # ==============================================
        # 情况 1：传入了新音频 → 先拼接视频+丢弃所有音频
        # ==============================================
        if audio and isinstance(audio, dict) and "waveform" in audio:
            temp_concat = os.path.join(output_dir, "_temp_no_audio.mp4")
            
            # 拼接视频，不带音频（超快）
            cmd_concat = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-f', 'concat', '-safe', '0', '-i', list_path,
                '-c:v', 'copy',  # 视频不编码
                '-an',           # 清空音频（最后统一加新的）
                '-movflags', '+faststart',
                temp_concat
            ]
            subprocess.run(cmd_concat, check=True, capture_output=True)

            # 统一加音频（只编码音频，视频不动）
            audio_wav = audio_tensor_to_wav_ffmpeg(audio)
            cmd_final = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', temp_concat,
                '-i', audio_wav,
                '-c:v', 'copy',
                '-c:a', 'aac', '-b:a', '192k', '-ac', '2',
                '-map', '0:v:0', '-map', '1:a:0',
                '-movflags', '+faststart',
                output_path
            ]
            subprocess.run(cmd_final, check=True, capture_output=True)

        # ==============================================
        # 情况 2：没传音频 → 视频+音频 直接一起拼接（完全不编码！）
        # ==============================================
        else:
            cmd_concat = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-f', 'concat', '-safe', '0', '-i', list_path,
                '-c', 'copy',  # 视频+音频全部直接复制，不编码
                '-movflags', '+faststart',
                output_path
            ]
            subprocess.run(cmd_concat, check=True, capture_output=True)

        return output_path

    except Exception as e:
        print(f"[凤希AI视频合并失败] {e}")
        return None

    finally:
        if list_path and os.path.exists(list_path):
            os.remove(list_path)
        if temp_concat and os.path.exists(temp_concat):
            os.remove(temp_concat)

# ==========================================================================

class FxAiVideoMerger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "源视频文件夹路径": ("STRING", {"default": ""}),
                "文件数量": ("INT", {"default": 1, "step": 1}),
                "名称前缀": ("STRING", {"default": "fxai_"}),
            },
            "optional": {
                "音频": ("AUDIO",),
                "刷新标记": ("ANY",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("视频本地路径",)
    FUNCTION = "run"
    CATEGORY = "凤希AI/视频"

    def run(self, 源视频文件夹路径, 文件数量=1, 名称前缀="fxai_", 音频=None, 刷新标记=None):
        time_str = time.strftime("%Y%m%d_%H%M%S")
        final_name = f"{名称前缀}{time_str}"
        
        if not os.path.isdir(源视频文件夹路径):
            return ("",)

        video_path = merge_videos(源视频文件夹路径, final_name, 文件数量, 音频)
        print(f"[凤希AI] ✅ 视频生成完毕。路径：{video_path}")        
        gc.collect()
        return (video_path or "",)