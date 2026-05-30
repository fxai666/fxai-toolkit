import os
import torch
import subprocess
import numpy as np
import json

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpeg', '.mpg', '.webm')

def frame_to_comfy_image(frame):
    frame = frame.astype(np.float32) / 255.0
    tensor = torch.from_numpy(frame)[None,]
    return tensor

def decode_video_frames(video_path, start_second, end_second):
    video_path = os.path.abspath(video_path)

    cmd_probe = [
        'ffprobe', video_path,
        '-hide_banner', '-loglevel', 'error',
        '-show_streams', '-select_streams', 'v:0',
        '-of', 'json'
    ]

    result = subprocess.run(
        cmd_probe,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=os.name == 'nt'
    )

    if not result.stdout:
        raise RuntimeError("FFprobe 无法读取视频")

    probe = json.loads(result.stdout)
    video_stream = probe['streams'][0]

    fps_str = video_stream.get('r_frame_rate', '0/1')
    num, den = map(int, fps_str.split('/'))
    fps = num / den if den != 0 else 30.0

    width = int(video_stream['width'])
    height = int(video_stream['height'])
    duration = float(video_stream.get('duration', 1))

    start = max(0.0, start_second)
    if end_second <= start or end_second == 0:
        end = duration
    else:
        end = min(end_second, duration)

    cmd_decode = [
        'ffmpeg',
        '-ss', str(start),
        '-i', video_path,
        '-to', str(end),
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-vsync', '0',
        '-hide_banner',
        '-loglevel', 'error',
        '-'
    ]

    process = subprocess.Popen(
        cmd_decode,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=os.name == 'nt'
    )

    frames = []
    frame_size = width * height * 3

    while True:
        buffer = process.stdout.read(frame_size)
        if not buffer:
            break
        frame = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)
        frames.append(frame)

    process.wait()
    return frames, fps, (height, width)

class FxAiVideoLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "视频文件夹路径": ("STRING", {"default": "", "multiline": False}),
                "视频索引": ("INT", {"default": 0, "min": 0}),
                "起始时间": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "结束时间": ("FLOAT", {"default": 0.0, "min": 0.0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("视频帧序列(视频格式)", "所有帧图片序列(图片格式)", "当前视频路径", "总帧数", "帧率(FPS)", "分辨率")
    FUNCTION = "load_video"
    CATEGORY = "凤希AI/视频"

    def load_video(self, 视频文件夹路径, 视频索引, 起始时间, 结束时间):
        folder = 视频文件夹路径.strip()
        if not os.path.isdir(folder):
            raise RuntimeError("文件夹不存在")

        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXTENSIONS)])
        if not files:
            raise RuntimeError("文件夹内无视频文件")

        if 视频索引 < 0 or 视频索引 >= len(files):
            raise RuntimeError("视频索引越界")

        video_path = os.path.join(folder, files[视频索引])
        frames_bgr, fps, (h, w) = decode_video_frames(video_path, 起始时间, 结束时间)

        if not frames_bgr:
            raise RuntimeError("未解码到任何视频帧")

        comfy_frames = [frame_to_comfy_image(f) for f in frames_bgr]
        video_tensor = torch.cat(comfy_frames, dim=0)
        all_frames_tensor = video_tensor
        total = len(comfy_frames)
        fps_int = round(fps) if fps > 0 else 30

        return (
            video_tensor,
            all_frames_tensor,
            video_path,
            total,
            fps_int,
            f"{w}x{h}"
        )