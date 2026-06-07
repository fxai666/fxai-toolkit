import os
import re
import time
import shutil
import subprocess
import torch
import numpy as np
import gc
import psutil
import ctypes
from ctypes import wintypes

import folder_paths
import comfy.model_management

# -----------------------------------------------------------------------------
# 🔥 终极杀进程：递归杀所有子/孙进程（ffmpeg 必死）
# -----------------------------------------------------------------------------
def kill_all_child_processes():
    try:
        current_pid = os.getpid()
        parent = psutil.Process(current_pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except Exception:
                pass
        time.sleep(0.1)
    except Exception:
        pass

# -----------------------------------------------------------------------------
# 🔥 Windows 强制释放物理内存+虚拟内存（强力版）
# -----------------------------------------------------------------------------
def win32_empty_working_set():
    if os.name != "nt":
        return
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        hProcess = kernel32.GetCurrentProcess()
        # 强制清空工作集
        psapi.EmptyWorkingSet(hProcess)
        # 极限压缩工作集
        kernel32.SetProcessWorkingSetSize(hProcess, wintypes.SIZE_T(-1), wintypes.SIZE_T(-1))
    except Exception:
        pass

# -----------------------------------------------------------------------------
# 🔥 终极内存清理：显存+内存+进程+Windows工作集
# -----------------------------------------------------------------------------
def safe_memory_clean():
    try:
        # 1. 释放显存
        with torch.no_grad():
            torch.clear_autocast_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        # 2. ComfyUI 缓存
        try:
            comfy.model_management.soft_empty_cache()
        except Exception:
            pass
        # 3. Python GC（一次到位）
        gc.collect()
        # 4. Windows 强制释放物理/虚拟内存
        win32_empty_working_set()
        # 5. 杀所有残留子进程
        kill_all_child_processes()
    except Exception as e:
        print("[凤希AI] 内存清理异常:", str(e))

# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------
def safe_path_join(base_dir, path):
    base_dir = os.path.abspath(base_dir).lower()
    full_path = os.path.abspath(os.path.join(base_dir, path)).lower()
    return full_path if full_path.startswith(base_dir) else None

def get_last_number(target_dir):
    used = set()
    if os.path.isdir(target_dir):
        for f in os.listdir(target_dir):
            m = re.match(r'^(\d+)', f)
            if m:
                used.add(int(m.group(1)))
    next_num = 0
    while next_num in used:
        next_num += 1
    return next_num

def get_video_dir(subdir=""):
    comfy_root = folder_paths.base_path
    base_dir = "fxai/video"
    target_dir = os.path.join(comfy_root, base_dir)
    if subdir:
        subdir = re.sub(r'[\\/*?:"<>|]', "", subdir)
        target_dir = os.path.join(target_dir, subdir)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

def get_fixed_temp_audio_path():
    comfy_root = folder_paths.base_path
    temp_dir = os.path.join(comfy_root, "fxai/video/temp")
    os.makedirs(temp_dir, exist_ok=True)
    return os.path.join(temp_dir, "fxai_temp_audio.wav")

# -----------------------------------------------------------------------------
# 音频转换（communicate 安全版，无管道泄漏）
# -----------------------------------------------------------------------------
def audio_tensor_to_wav_ffmpeg(audio_dict):
    temp_path = ""
    proc = None
    waveform = None
    waveform_np = None
    audio_data = None
    raw_pcm = None

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
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'quiet', '-nostats',
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
            stderr=subprocess.DEVNULL,
            close_fds=True
        )
        # ✅ 一次性写入+等待，不残留
        proc.communicate(input=raw_pcm, timeout=300)

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

        return temp_path

    except Exception as e:
        print("[凤希AI FFmpeg音频转换失败]", str(e))
        return ""

    finally:
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=3)
            except Exception:
                pass
        del waveform, waveform_np, audio_data, raw_pcm
        safe_memory_clean()

# -----------------------------------------------------------------------------
# 视频生成（流式写入，communicate 安全）
# -----------------------------------------------------------------------------
def save_video(images, save_dir, fps=24, custom_num=0, audio=None, transition_frames=1):
    safe_memory_clean()
    proc = None
    img_np = None
    batch_data = None

    try:
        num = custom_num if custom_num >= 0 else get_last_number(save_dir)
        filename = f"{num:03d}.mp4"
        save_path = safe_path_join(save_dir, filename)

        img_np = (images.cpu().numpy() * 255).astype(np.uint8)
        del images
        safe_memory_clean()

        total_len = img_np.shape[0]
        img_np = img_np[: total_len - transition_frames]

        if len(img_np) == 0:
            print("[凤希AI视频合成失败] 没有有效帧")
            return ""

        height, width = img_np[0].shape[0], img_np[0].shape[1]

        if isinstance(audio, dict) and "waveform" in audio:
            audio = audio_tensor_to_wav_ffmpeg(audio)

        if audio and os.path.exists(audio):
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'quiet', '-nostats',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'rgb24',
                '-r', str(fps),
                '-i', '-',
                '-i', audio,
                '-c:v', 'libx264',
                '-preset', 'slow',
                '-crf', '17',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',
                '-movflags', '+faststart',
                save_path
            ]
        else:
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'quiet', '-nostats',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'rgb24',
                '-r', str(fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-crf', '17',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                save_path
            ]

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            bufsize=1024*1024*10
        )

        batch_size = 20
        for i in range(0, len(img_np), batch_size):
            batch = img_np[i:i+batch_size]
            batch_data = b''.join([img.tobytes() for img in batch])
            try:
                proc.stdin.write(batch_data)
            except BrokenPipeError:
                print("[凤希AI] FFmpeg管道断开，停止写入")
                break
            except Exception as e:
                print("[凤希AI] 视频帧写入失败:", str(e))
                break

        proc.stdin.close()
        proc.wait(timeout=180)

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

        print("[凤希AI视频] 成功保存：", save_path)
        return save_path

    except Exception as e:
        print("[凤希AI视频合成失败]", str(e))
        return ""

    finally:
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=3)
            except Exception:
                pass
        if img_np is not None:
            del img_np
        if batch_data is not None:
            del batch_data
        safe_memory_clean()

# -----------------------------------------------------------------------------
# 节点主类
# -----------------------------------------------------------------------------
class FxAiVideoGeneratorV3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片序列": ("IMAGE",),
                "目录": ("STRING", {"default": "sucai"}),
                "帧率FPS": ("INT", {"default": 24, "min": 1}),
                "视频序号": ("INT", {"default": 0, "min": 0}),
            },
            "optional": {
                "音频": ("AUDIO",),
                "过渡帧数": ("INT", {"default": 1, "min": 1}),
                "过渡帧引导": ("IMAGE",),
                "视频帧序列": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT")
    RETURN_NAMES = ("过渡帧", "视频文件路径", "保存目录", "实际帧数")
    FUNCTION = "run"
    CATEGORY = "凤希AI/视频"

    def run(self, 目录, 帧率FPS, 视频序号, 图片序列, 音频=None, 过渡帧数=1, 过渡帧引导=None, 视频帧序列=None):
        if 图片序列 is None and 视频帧序列 is None:
            return (图片序列, "", "", 0)

        target_dir = get_video_dir(目录)

        total_frames = len(图片序列)
        actual_frames = total_frames - 过渡帧数

        transition_frames_out = 图片序列[-过渡帧数:]

        if 过渡帧引导 is not None and len(过渡帧引导) > 0:
            guide_frame = 过渡帧引导[0:1]
            transition_frames_out = torch.cat([transition_frames_out, guide_frame], dim=0)

        if 视频帧序列 is not None and len(视频帧序列) > 0:
            video_frames = 视频帧序列
        else:
            video_frames = 图片序列

        video_path = save_video(
            images=video_frames,
            save_dir=target_dir,
            fps=帧率FPS,
            custom_num=视频序号,
            audio=音频,
            transition_frames=过渡帧数
        )

        del 图片序列, video_frames
        safe_memory_clean()

        return (transition_frames_out, video_path, target_dir, actual_frames)