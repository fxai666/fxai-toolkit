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

# 安全路径校验
def safe_path_join(base_dir, path):
    base_dir = os.path.abspath(base_dir)
    full_path = os.path.abspath(os.path.join(base_dir, path))
    if not full_path.startswith(base_dir):
        return None
    return full_path

# 获取下一个编号
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

# 音频张量转WAV（不变）
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
        proc.stdin.write(raw_pcm)
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

# 视频合成：使用rawvideo管道 + 分批写入（最快+最高质量+低内存）
def save_video(images, save_dir, fps=24, custom_num=0, audio=""):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    num = custom_num if custom_num >= 0 else get_last_number(save_dir)
    filename = f"{num:03d}.mp4"
    save_path = safe_path_join(save_dir, filename)

    # 转numpy + 移除最后一帧
    img_np = (images.cpu().numpy() * 255).astype(np.uint8)
    img_np = img_np[:-1]

    if len(img_np) == 0:
        print("[凤希AI视频合成失败] 没有有效帧")
        return ""

    try:
        height, width = img_np[0].shape[0], img_np[0].shape[1]
        
        # 音频处理
        if isinstance(audio, dict) and "waveform" in audio:
            audio = audio_tensor_to_wav_ffmpeg(audio)

        # 构建ffmpeg命令
        if audio and os.path.exists(audio):
            cmd = [
                'ffmpeg', '-y',
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
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'rgb24',
                '-r', str(fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-preset', 'slow',
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
            bufsize=1024*1024*10  # 10MB缓冲区
        )

        # 分批写入，避免内存峰值过高（每批20帧）
        batch_size = 20
        for i in range(0, len(img_np), batch_size):
            batch = img_np[i:i+batch_size]
            batch_data = b''.join([img.tobytes() for img in batch])
            proc.stdin.write(batch_data)
            # batch_data 在此释放，batch 在下一轮覆盖
        
        proc.stdin.close()
        proc.wait()

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

    except Exception as e:
        print(f"[凤希AI视频合成失败] {str(e)}")
        import traceback
        traceback.print_exc()
        return ""

    print(f"[凤希AI视频] 成功保存：{save_path}")
    return save_path

class FxAiVideoGenerator:
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
            }
        }

    RETURN_TYPES = ("IMAGE","STRING", "STRING")
    RETURN_NAMES = ("图片序列", "视频文件路径", "保存目录")
    FUNCTION = "run"
    CATEGORY = "凤希AI/视频"

    def run(self, 目录, 帧率FPS, 视频序号, 图片序列, 音频=""):
        if 图片序列 is None:
            return (图片序列, "", "")
        
        target_dir = get_video_dir(目录)
        video_path = save_video(
            images=图片序列,
            save_dir=target_dir,
            fps=帧率FPS,
            custom_num=视频序号,
            audio=音频
        )
        
        return (图片序列, video_path, target_dir)