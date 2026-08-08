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
    target_dir = os.path.join(comfy_root, "fxai", "video", "merged")
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

def get_fixed_temp_audio_path():
    comfy_root = folder_paths.base_path
    temp_dir = os.path.join(comfy_root, "fxai", "video", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    return os.path.join(temp_dir, "fxai_merge_temp_audio.wav")

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
        if os.path.exists(pcm_temp):
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
# 替换视频音频
# --------------------------
def replace_video_audio(video_path, audio_path):
    if not os.path.exists(video_path) or not os.path.exists(audio_path):
        return video_path

    base, ext = os.path.splitext(video_path)
    temp_video = f"{base}_temp{ext}"
    if os.path.exists(temp_video):
        os.remove(temp_video)
        
    try:
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', video_path, '-i', audio_path,
            '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '192k', '-ac', '2',
            '-map', '0:v:0', '-map', '1:a:0', '-shortest',
            '-movflags', '+faststart',
            temp_video
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)
        shutil.copy2(temp_video, video_path)
    except Exception as e:
        print(f"[凤希AI音频替换失败] {e}")
    finally:
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
# 视频合并主函数
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
        output_path = os.path.join(output_dir, f"{output_name}.mp4")

        if not videos:
            print("[凤希AI视频合并] 未找到视频文件")
            return None

        # 生成拼接列表
        list_path = os.path.join(source_dir, "merge_list.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for p in videos:
                # 统一路径分隔符，避免Windows报错
                safe_path = p.replace("\\", "/")
                f.write(f"file '{safe_path}'\n")

        # 带音频合并
        if audio and isinstance(audio, dict) and "waveform" in audio:
            temp_concat = os.path.join(output_dir, "_temp_no_audio.mp4")
            
            cmd_concat = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-f', 'concat', '-safe', '0', '-i', list_path,
                '-c:v', 'copy',
                '-an',
                '-movflags', '+faststart',
                temp_concat
            ]
            subprocess.run(cmd_concat, check=True, capture_output=True, text=True)

            audio_wav = audio_tensor_to_wav_ffmpeg(audio)
            if not audio_wav or not os.path.exists(audio_wav):
                return None

            cmd_final = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', temp_concat,
                '-i', audio_wav,
                '-c:v', 'copy',
                '-c:a', 'aac', '-b:a', '192k', '-ac', '2',
                '-map', '0:v:0', '-map', '1:a:0',
                '-shortest',
                '-movflags', '+faststart',
                output_path
            ]
            subprocess.run(cmd_final, check=True, capture_output=True, text=True)

        # 无音频直接拼接
        else:
            cmd_concat = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-f', 'concat', '-safe', '0', '-i', list_path,
                '-c', 'copy',
                '-movflags', '+faststart',
                output_path
            ]
            subprocess.run(cmd_concat, check=True, capture_output=True, text=True)

        return output_path

    except Exception as e:
        print(f"[凤希AI视频合并失败] {str(e)}")
        return None

    finally:
        # 清理临时文件
        for f in [list_path, temp_concat]:
            if f and os.path.exists(f):
                os.remove(f)
        gc.collect()

# ==========================================================================

class FxAiVideoMerger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "源视频文件夹路径": ("STRING", {"default": ""}),
                "文件数量": ("INT", {"default": 1, "min": 1, "step": 1}),
                "名称前缀": ("STRING", {"default": "fxai"}),
            },
            "optional": {
			    "刷新标记": ("INT", {"forceInput": True}),
                "音频": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("视频本地路径",)
    FUNCTION = "run"
    CATEGORY = "凤希AI/视频"

    def run(self, 源视频文件夹路径, 文件数量=1, 名称前缀="fxai",刷新标记=0, 音频=None):
        time_str = time.strftime("%Y%m%d_%H%M%S")
        final_name = f"{名称前缀}_{time_str}"
        
        if not os.path.isdir(源视频文件夹路径):
            print("[凤希AI] 错误：路径不是文件夹")
            return ("",)

        video_path = merge_videos(源视频文件夹路径, final_name, 文件数量, 音频)
        if video_path and os.path.exists(video_path):
            print(f"[凤希AI] ✅ 视频合并完成：{video_path}")
        else:
            print("[凤希AI] ❌ 视频合并失败")
            
        gc.collect()
        return (video_path or "",)
		
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")