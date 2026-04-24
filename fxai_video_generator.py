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

# 音频张量转临时WAV
def audio_tensor_to_wav(audio_dict):
    try:
        waveform = audio_dict["waveform"]
        sample_rate = audio_dict["sample_rate"]
        waveform = waveform.squeeze(0)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.close()
        temp_path = temp_file.name

        import torchaudio
        torchaudio.save(
            temp_path,
            waveform.cpu(),
            sample_rate,
            format="wav",
            encoding="PCM_S",
            bits_per_sample=16
        )
        return temp_path
    except Exception as e:
        print(f"[凤希音频转换失败] {str(e)}")
        return ""

# 图片+音频合成视频
def save_video(images, save_dir, fps=24, custom_num=0, audio_path=""):
    num = custom_num if custom_num >= 0 else get_last_number(save_dir)
    filename = f"{num:03d}.mp4"
    save_path = safe_path_join(save_dir, filename)

    # ✅ 全局固定 frame 文件夹：fxai/video/frame
    temp_frames = get_global_frame_dir()

    # 直接覆盖写入，纯数字命名：0001.png / 0002.png
    img_np = (images.cpu().numpy() * 255).astype(np.uint8)
    for i in range(img_np.shape[0]):
        Image.fromarray(img_np[i]).save(os.path.join(temp_frames, f"{i:04d}.png"))

    temp_audio = ""
    try:
        if isinstance(audio_path, dict) and "waveform" in audio_path:
            temp_audio = audio_tensor_to_wav(audio_path)
            audio_path = temp_audio

        # FFmpeg 读取全局帧
        if audio_path and os.path.exists(audio_path):
            cmd = f'ffmpeg -y -framerate {fps} -i "{temp_frames}/%04d.png" -i "{audio_path}" -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "{save_path}"'
        else:
            cmd = f'ffmpeg -y -framerate {fps} -i "{temp_frames}/%04d.png" -c:v libx264 -pix_fmt yuv420p "{save_path}"'

        subprocess.run(cmd, shell=True, check=True, capture_output=True)
    finally:
        if temp_audio and os.path.exists(temp_audio):
            os.unlink(temp_audio)

    print(f"[凤希视频] 成功保存：{save_path}")
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
            return ("未输入图片序列", "")
        
        target_dir = get_video_dir(目录)
        video_path = save_video(
            images=图片序列,
            save_dir=target_dir,
            fps=帧率FPS,
            custom_num=视频序号,
            audio_path=音频
        )
        return (图片序列, video_path, target_dir)