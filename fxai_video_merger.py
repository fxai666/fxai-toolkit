import os
import re
import time
import shutil
import folder_paths
import subprocess
import torch
import numpy as np
import gc
import psutil

import comfy.model_management

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

def audio_tensor_to_wav_ffmpeg(audio_dict):
    proc = None
    waveform = None
    waveform_np = None
    audio_data = None
    raw_pcm = None
    temp_path = ""

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
        raw_pcm = audio_data.tobytes()

        temp_path = get_fixed_temp_audio_path()

        cmd = [
            'ffmpeg', '-y',
            '-hide_banner', '-loglevel', 'quiet', '-nostats',
            '-f', 'f32le',
            '-ar', str(sample_rate),
            '-ac', '2',
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
        proc.stdin.write(raw_pcm)
        proc.stdin.close()
        proc.wait()

        return temp_path if proc.returncode == 0 else ""

    except Exception as e:
        print(f"[凤希AI音频转换失败] {e}")
        return ""

    finally:
        if proc is not None:
            try:
                proc.stdin.close()
            except:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                pass

        del waveform, waveform_np, audio_data, raw_pcm
        gc.collect()

def replace_video_audio(video_path, audio_path):
    if not os.path.exists(video_path) or not os.path.exists(audio_path):
        return video_path

    temp_video = video_path.replace(".mp4", "_temp.mp4")
    try:
        cmd = (
            f'ffmpeg -y -hide_banner -loglevel quiet -nostats -i "{video_path}" -i "{audio_path}" '
            f'-c:v copy -c:a aac -ac 2 -map 0:v:0 -map 1:a:0 -shortest "{temp_video}"'
        )
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        shutil.move(temp_video, video_path)
    except Exception as e:
        # print(f"[凤希AI替换音频失败] {e}")
        if os.path.exists(temp_video):
            os.remove(temp_video)
    return video_path

def get_video_files(source_dir, max_count=0):
    if not os.path.isdir(source_dir):
        return []
    
    if max_count is None:
        max_count = 0
        
    exts = ('.mp4', '.webm', '.mov', '.avi')
    files = sorted(f for f in os.listdir(source_dir) if f.lower().endswith(exts))
    if max_count > 0:
        files = files[:max_count]
    return [safe_path_join(source_dir, f) for f in files]

def merge_videos(source_dir, output_name, max_count=0, audio=None):
    videos = []
    list_path = None
    output_path = None
    audio_wav = None

    try:
        videos = get_video_files(source_dir, max_count)
        output_dir = get_merge_output_dir()
        output_name = re.sub(r'[\\/*?:"<>|]', "", output_name.strip())
        output_path = safe_path_join(output_dir, f"{output_name}.mp4")

        if not videos:
            print("[凤希AI视频合并] 无视频")
            return None
        
        elif len(videos) == 1:
            shutil.copy2(videos[0], output_path)
        
        else:
            list_path = os.path.join(source_dir, "merge_list.txt")
            with open(list_path, "w", encoding="utf-8") as f:
                for p in videos:
                    f.write(f"file '{p}'\n")
                    
            cmd = f'ffmpeg -y -hide_banner -loglevel quiet -nostats -f concat -safe 0 -i "{list_path}" -c copy "{output_path}"'
            subprocess.run(cmd, shell=True, check=True)

        # 音频处理
        if audio and isinstance(audio, dict) and "waveform" in audio:
            audio_wav = audio_tensor_to_wav_ffmpeg(audio)
            if audio_wav:
                replace_video_audio(output_path, audio_wav)

        return output_path

    except Exception as e:
        print(f"[凤希AI视频合并失败] {e}")
        return None

    finally:
        if list_path and os.path.exists(list_path):
            os.remove(list_path)

        gc.collect()

def release_all_resources():
    try:
        print("[凤希AI] 开始彻底释放内存/显存/进程资源...")

        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # 清理子进程
        try:
            current_pid = os.getpid()
            process = psutil.Process(current_pid)
            children = process.children(recursive=True)
            for child in children:
                try:
                    child.kill()
                except:
                    pass
        except:
            pass

        gc.collect()
        print("[凤希AI] 所有资源已完全释放 ✅")

    except Exception as e:
        print(f"[凤希AI] 资源释放失败：{str(e)}")

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
        print(f"[凤希AI] 视频生成完毕。")
        
        release_all_resources()
        return (video_path or "",)