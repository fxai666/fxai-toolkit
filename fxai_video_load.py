import os
import torch
import cv2
import ffmpeg
import numpy as np
from PIL import Image
from typing import Tuple, List

# 支持的视频格式
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpeg', '.mpg', '.webm')

def frame_to_comfy_image(frame: np.ndarray) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(frame_rgb)[None,]
    return tensor

def decode_video_frames(
    video_path: str,
    start_second: float = 0.0,
    end_second: float = 0.0
) -> Tuple[List[np.ndarray], float, Tuple[int, int]]:
    probe = ffmpeg.probe(video_path)
    video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
    if not video_stream:
        raise RuntimeError("无视频流")

    fps = eval(video_stream['r_frame_rate'])
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    duration = float(video_stream.get('duration', 0))

    start = max(0.0, start_second)
    if end_second <= start or end_second == 0:
        end = duration
    else:
        end = min(end_second, duration)

    cmd = (
        ffmpeg
        .input(video_path, ss=start, to=end)
        .output('pipe:', format='rawvideo', pix_fmt='bgr24', vsync='0')
        .global_args('-hide_banner', '-loglevel', 'error')
        .run_async(pipe_stdout=True)
    )

    frames = []
    frame_size = height * width * 3
    while True:
        b = cmd.stdout.read(frame_size)
        if not b: break
        frame = np.frombuffer(b, dtype=np.uint8).reshape((height, width, 3))
        frames.append(frame)
    cmd.wait()
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

    # 输出类型：帧率改为 INT 整数
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("视频帧序列(视频格式)", "所有帧图片序列(图片格式)", "当前视频路径", "总帧数", "帧率(FPS)", "分辨率")
    FUNCTION = "load_video"
    CATEGORY = "凤希AI/视频"

    def load_video(self, 视频文件夹路径, 视频索引, 起始时间, 结束时间):
        folder = 视频文件夹路径.strip()

        if not os.path.isdir(folder):
            raise RuntimeError(f"文件夹不存在: {folder}")

        # 遍历文件夹 → 按文件名排序取视频
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXTENSIONS)])
        if not files:
            raise RuntimeError("文件夹内无视频文件")

        # 索引越界处理
        if 视频索引 < 0 or 视频索引 >= len(files):
            raise RuntimeError(f"索引越界，共 {len(files)} 个视频，索引只能 0~{len(files)-1}")

        # 视频路径
        video_path = os.path.join(folder, files[视频索引])

        # 解码帧
        frames_bgr, fps, (h, w) = decode_video_frames(video_path, 起始时间, 结束时间)

        # 转格式
        comfy_frames = [frame_to_comfy_image(f) for f in frames_bgr]
        video_tensor = torch.cat(comfy_frames, dim=0)
        all_frames_tensor = video_tensor  # 完整图片序列
        total = len(comfy_frames)
        
        # 帧率 四舍五入转整数
        fps_int = round(fps)

        return (
            video_tensor,
            all_frames_tensor,
            video_path,
            total,
            fps_int,  # 输出整数帧率
            f"{w}x{h}"
        )