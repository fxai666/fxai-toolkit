import os
import torch
import subprocess
import numpy as np
import json

import folder_paths

from fxai_sort_utils import natural_key as _natural_key
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpeg', '.mpg', '.webm')
AUDIO_SAMPLE_RATE = 44100


def expand_path(target):
    """相对写法展开成绝对路径，绝对路径原样返回。

    / 开头 → 相对 ComfyUI 根（/fxai/video/a.mp4 = <base>/fxai/video/a.mp4）。
    Windows 上 os.path.isabs("/x") 是 True，所以这个分支必须放在 isabs 前面。
    其余相对写法 → 先按 input 目录找，找不到保持原样，交给下面的报错。
    """
    if target.startswith(("/", "\\")):
        return os.path.join(folder_paths.base_path, target.lstrip("/\\"))
    if not os.path.isabs(target):
        candidate = os.path.join(folder_paths.get_input_directory(), target)
        if os.path.exists(candidate):
            return candidate
    return target


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
    # 结束时间是「到第几秒」，不是「跑多久」。-ss 放在 -i 前面时，-to 是相对
    # 定位点的，所以要自己换算成 -t 的时长，否则从第 10 秒起跑 10~20 会出 20 秒。
    if end_second <= start or end_second == 0:
        length = 0.0
    else:
        length = min(end_second, duration) - start

    cmd_decode = [
        'ffmpeg',
        '-ss', str(start),
        '-i', video_path,
    ]
    if length > 0:
        cmd_decode += ['-t', str(length)]
    cmd_decode += [
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

def decode_audio(video_path, start_second, end_second):
    """抽同一段视频的音轨，返回 ComfyUI 的 AUDIO 结构；没有音轨返回 None。

    音轨不走视频那条 rawvideo 管道，单独跑一次 ffmpeg 直接吐 f32le：
    比"先落一个 wav 再读回来"少一次磁盘往返，也不用额外依赖。
    起止时间和画面用同一套切法，两边的段长才会一致，配音不会错位。
    """
    video_path = os.path.abspath(video_path)
    start = max(0.0, start_second)

    cmd = ['ffmpeg', '-ss', str(start), '-i', video_path]
    if end_second > start:
        cmd += ['-t', str(end_second - start)]
    cmd += [
        '-vn',
        '-f', 'f32le',
        '-ar', str(AUDIO_SAMPLE_RATE),
        '-ac', '2',
        '-hide_banner', '-loglevel', 'error',
        '-'
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=os.name == 'nt'
    )
    raw, _ = process.communicate()

    if not raw:
        print(f"[凤希AI] 视频没有音轨，音频输出为空：{video_path}")
        return None

    samples = np.frombuffer(raw, dtype=np.float32)
    samples = samples[:samples.size - samples.size % 2].reshape(-1, 2)
    waveform = torch.from_numpy(np.ascontiguousarray(samples.T)).unsqueeze(0).float()
    return {"waveform": waveform, "sample_rate": AUDIO_SAMPLE_RATE}


class FxAiVideoLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "视频文件夹路径": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "两种填法都行：\n"
                               "① 填一个**目录** → 按 视频索引 挑这个目录里排序后的第几个视频；\n"
                               "② 直接填一个**视频文件的完整路径**（如 input/xxx.mp4）\n"
                               "   → 只读这一个文件，视频索引 自动失效。\n"
                               "填文件时不怕目录里增删文件、也不怕改名排序变化，最稳。"}),
                "视频索引": ("INT", {
                    "default": -1, "min": -1,
                    "tooltip": "**-1（默认）= 按上面填的实际路径读**，索引不参与；\n"
                               "填的是视频文件时，永远是读这个文件；\n"
                               "填的是目录时，-1 就等于取第一个（索引 0）。\n"
                               "0 及以上 = 按目录里文件名排序后的第几个视频挑（从 0 开始）。"}),
                "起始时间": ("FLOAT", {
                    "default": 0.0, "min": 0.0,
                    "tooltip": "从第几秒开始读，0 = 从头。"}),
                "结束时间": ("FLOAT", {
                    "default": 0.0, "min": 0.0,
                    "tooltip": "读到第几秒（不是读多久），0 = 读到结尾。\n"
                               "例：10 ~ 20 = 只读第 10 到 20 秒那一段。"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "INT", "INT", "STRING", "AUDIO", "INT", "INT")
    RETURN_NAMES = ("视频帧序列(视频格式)", "所有帧图片序列(图片格式)", "当前视频路径", "总帧数", "帧率(FPS)", "分辨率", "音频",
                    "视频宽度", "视频高度")
    FUNCTION = "load_video"
    CATEGORY = "凤希AI/视频"

    @staticmethod
    def resolve_video_path(路径, 索引=-1):
        """把「实际视频路径」或「目录 + 索引」统一成一个视频文件路径。

        一个参数两种用法，是为了不新增输入（新增 widget 会让老工作流的
        widgets_values 整体错位）。

        规则（和面板提示一致）：
            索引 = -1（默认）→ 路径模式；填目录时退化成取第一个
            索引 >= 0       → 目录模式，按文件名排序取第 N 个
        文件永远优先：填的就是一个视频文件时，索引填几都是读它。
        """
        target = str(路径 or "").strip().strip('"').strip("'")
        if not target:
            raise RuntimeError(
                "「视频文件夹路径」没填。可以填一个目录，"
                "也可以直接填一个视频文件的完整路径（如 input/a.mp4）")

        target = expand_path(target)

        try:
            idx = int(索引)
        except Exception:
            idx = -1

        if os.path.isfile(target):
            if not target.lower().endswith(VIDEO_EXTENSIONS):
                raise RuntimeError(
                    f"这个文件不是视频：{target}\n"
                    f"支持的扩展名：{' '.join(VIDEO_EXTENSIONS)}")
            return target

        if os.path.isdir(target):
            files = sorted([f for f in os.listdir(target)
                            if f.lower().endswith(VIDEO_EXTENSIONS)], key=_natural_key)
            if not files:
                raise RuntimeError(f"文件夹内无视频文件：{target}")
            use = idx if idx >= 0 else 0      # -1 = 没指定索引，取第一个
            if use >= len(files):
                raise RuntimeError(
                    f"视频索引越界：索引 {use}，该目录共 {len(files)} 个视频\n"
                    f"目录：{target}")
            return os.path.join(target, files[use])

        raise RuntimeError(
            f"路径不存在：{target}\n"
            "「视频文件夹路径」既可以填目录，也可以直接填视频文件的完整路径。")

    def load_video(self, 视频文件夹路径, 视频索引, 起始时间, 结束时间):
        # 填的是视频文件 → 直接用；填的是目录 → 用 视频索引 挑
        video_path = self.resolve_video_path(视频文件夹路径, 视频索引)

        frames_bgr, fps, (h, w) = decode_video_frames(video_path, 起始时间, 结束时间)

        if not frames_bgr:
            raise RuntimeError("未解码到任何视频帧")

        comfy_frames = [frame_to_comfy_image(f) for f in frames_bgr]
        video_tensor = torch.cat(comfy_frames, dim=0)
        all_frames_tensor = video_tensor
        total = len(comfy_frames)
        audio = decode_audio(video_path, 起始时间, 结束时间)

        return (
            video_tensor,
            all_frames_tensor,
            video_path,
            total,
            int(round(float(fps))) if fps > 0 else 30,
            f"{w}x{h}",
            audio,
            int(w),
            int(h),
        )