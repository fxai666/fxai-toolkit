import os
import re
import torch
import numpy as np
from PIL import Image
import folder_paths
import subprocess
import tempfile

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

# 全局固定的帧目录（永远在这里）
def get_global_frame_dir():
    comfy_root = folder_paths.base_path
    frame_dir = os.path.join(comfy_root, "fxai/video/frame")
    os.makedirs(frame_dir, exist_ok=True)
    return frame_dir

# ===================== 固定临时音频（覆盖写入，不删除） =====================
def get_fixed_temp_audio_path():
    comfy_root = folder_paths.base_path
    temp_dir = os.path.join(comfy_root, "fxai/video/temp")
    os.makedirs(temp_dir, exist_ok=True)
    return os.path.join(temp_dir, "fxai_temp_audio.wav")

# ===================== ✅ 新版：用 FFmpeg 把音频张量保存为 WAV =====================
def audio_tensor_to_wav_ffmpeg(audio_dict):
    try:
        waveform = audio_dict["waveform"]  # [1, 1, T] 或 [1, T]
        sample_rate = audio_dict["sample_rate"]
        
        # 降维成 FFmpeg 能用的格式
        waveform = waveform.squeeze().cpu().numpy()  # shape: [T] 或 [2, T]
        temp_path = get_fixed_temp_audio_path()

        # 音频是 float32 张量 → 写入原始二进制，再用 FFmpeg 转 WAV
        raw_pcm = waveform.astype(np.float32).tobytes()

        # 判定声道数
        channels = 1 if waveform.ndim == 1 else waveform.shape[0]

        # FFmpeg 从原始 PCM 生成标准 WAV
        cmd = (
            f'ffmpeg -y -f f32le -ar {sample_rate} -ac {channels} -i pipe: '
            f'-c:a pcm_s16le "{temp_path}"'
        )

        # 直接喂二进制数据给 FFmpeg
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        proc.stdin.write(raw_pcm)
        proc.stdin.close()
        proc.wait()

        return temp_path
    except Exception as e:
        print(f"[凤希AI FFmpeg音频转换失败] {str(e)}")
        return ""

# 图片+音频合成视频
def save_video(images, save_dir, fps=24, custom_num=0, audio=""):
    num = custom_num if custom_num >= 0 else get_last_number(save_dir)
    filename = f"{num:03d}.mp4"
    save_path = safe_path_join(save_dir, filename)

    temp_frames = get_global_frame_dir()

    # 覆盖写入帧
    img_np = (images.cpu().numpy() * 255).astype(np.uint8)
    
    # ===================== ✅ 关键修改：去掉最后一帧 =====================
    img_np = img_np[:-1]  # 移除最后一张图片，留给下一个循环作为首帧
    # ==================================================================

    for i in range(img_np.shape[0]):
        Image.fromarray(img_np[i]).save(os.path.join(temp_frames, f"{i:04d}.png"))

    try:
        # ✅ 使用新版 FFmpeg 音频处理
        if isinstance(audio, dict) and "waveform" in audio:
            audio = audio_tensor_to_wav_ffmpeg(audio)

        # FFmpeg 合成
        if audio and os.path.exists(audio):
            cmd = f'ffmpeg -y -framerate {fps} -i "{temp_frames}/%04d.png" -i "{audio}" -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "{save_path}"'
        else:
            cmd = f'ffmpeg -y -framerate {fps} -i "{temp_frames}/%04d.png" -c:v libx264 -pix_fmt yuv420p "{save_path}"'

        subprocess.run(cmd, shell=True, check=True, capture_output=True)
    except Exception as e:
        print(f"[凤希AI视频合成失败] {str(e)}")
        return ""

    print(f"[凤希AI视频] 成功保存：{save_path}")
    return save_path

# 节点主体
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
    CATEGORY = "凤希AI"

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