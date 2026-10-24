# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# 视频分段用的公共层：探针、按帧解码、按帧切段。
# 「视频分段器V1」和「视频分段加载」共用这一份，段边界只在这里算一次，
# 两边不会各自取整跑偏。

import hashlib
import json
import math
import os
import subprocess

import numpy as np
import torch
import folder_paths

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpeg', '.mpg', '.webm')
H3_FPS = 24
AUDIO_SAMPLE_RATE = 44100


def align_down_h3(frames):
    """最大的 17k+5 <= frames（MiniMax H3 合法帧数网格）。"""
    if frames < 5:
        return 5
    return 17 * ((frames - 5) // 17) + 5


def align_down_ltx(frames):
    """最大的 8k+1 <= frames（LTX latent 时间步长 8，length 下限 9）。"""
    if frames <= 9:
        return 9
    return 8 * ((frames - 1) // 8) + 1


def round_half_up(value):
    """四舍五入。

    Python 内置 round 是银行家舍入（round(282.5) = 282），面板那边 JS 的
    Math.round 是往上进（283）。30fps 源做 MiniMax 分段时 226/24*30 正好
    落在 .5 上，两边会差一帧，所以这里统一成和 JS 一致的算法。
    """
    return int(math.floor(float(value) + 0.5))


def segment_frames(avg_dur, fps, model):
    """每段帧数。

    标准      平均分段时长 × 源帧率，四舍五入到帧
    LTX       同上，再向下对齐到 8k+1
    MiniMax   按 H3 固定 24fps 折算成 17k+5，再换回源帧率
    """
    name = str(model or "").strip().lower()
    if name in ("minimax", "minimax h3", "h3"):
        h3_frames = align_down_h3(round_half_up(avg_dur * H3_FPS))
        return max(1, round_half_up(h3_frames / H3_FPS * fps))
    if name == "ltx":
        return max(1, align_down_ltx(round_half_up(avg_dur * fps)))
    return max(1, round_half_up(avg_dur * fps))


def parse_keyframe_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        text = str(value).strip()
        if not text:
            return []
        parsed = json.loads(text)
        raw = parsed.get("keyframes", []) if isinstance(parsed, dict) else parsed
    return [max(0.0, float(x)) for x in raw]


def build_frame_segments(total_frames, fps, keyframes, skip_initial, include_tail,
                         avg_split, avg_dur, model):
    """返回 [(start_frame, frame_count), ...]。

    段与段首尾相接：下一段的 start 永远等于上一段的 start+count。
    循环取完既不漏帧，也不会把上一段的尾帧再当成下一段的首帧。
    """
    fps = max(float(fps), 1e-6)
    total_frames = max(1, int(total_frames))

    # 关键帧 → 帧号边界，落在两端之外的丢弃
    markers = []
    seen = set()
    for sec in keyframes or []:
        frame = round_half_up(max(0.0, float(sec)) * fps)
        if 0 < frame < total_frames and frame not in seen:
            seen.add(frame)
            markers.append(frame)
    markers.sort()

    points = [0] + markers + [total_frames]
    segments = [(points[i], points[i + 1] - points[i]) for i in range(len(points) - 1)]

    if skip_initial and len(segments) > 1:
        segments = segments[1:]
    if not include_tail and len(segments) > 1:
        segments = segments[:-1]

    min_frames = max(1, round_half_up(0.1 * fps))
    segments = [seg for seg in segments if seg[1] >= min_frames] or [(0, total_frames)]

    if not (avg_split and avg_dur > 0):
        return segments

    # 平均分段只在「跳过首段/去掉尾段」之后的区间里重新等分
    span_start = segments[0][0]
    span_end = segments[-1][0] + segments[-1][1]
    step = segment_frames(avg_dur, fps, model)

    out = []
    cursor = span_start
    while cursor < span_end:
        take = min(step, span_end - cursor)
        out.append((cursor, take))
        cursor += take

    # 末段太短并入前一段（阈值和音频分段器V2一致：>10 秒按 5 秒，否则 3 秒）
    threshold = (5.0 if avg_dur > 10.0 else 3.0) * fps
    if len(out) >= 2 and out[-1][1] <= threshold:
        prev_start, prev_count = out[-2]
        out[-2] = (prev_start, prev_count + out[-1][1])
        out.pop()
    return out


def list_input_video_files():
    """input 目录下的视频，返回相对路径（子目录/文件名），供面板和 API 用。"""
    input_dir = folder_paths.get_input_directory()
    if not input_dir or not os.path.isdir(input_dir):
        return []
    discovered = []
    for root, _dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(VIDEO_EXTENSIONS):
                full_path = os.path.join(root, filename)
                discovered.append(os.path.relpath(full_path, input_dir).replace("\\", "/"))
    return sorted(discovered)


def rel_video_path(video_path):
    """绝对路径 → 相对 input 目录的写法；不在 input 下就原样返回。

    分段器输出这个值，工作流换台机器跑时路径照样成立。
    """
    input_dir = folder_paths.get_input_directory()
    if not input_dir:
        return video_path
    try:
        rel = os.path.relpath(os.path.abspath(video_path), os.path.abspath(input_dir))
    except ValueError:
        return video_path
    if rel.startswith(".."):
        return video_path
    return rel.replace("\\", "/")


def resolve_video_path(raw):
    """相对 input 的「子目录/文件名」、/开头（相对 ComfyUI 根）、绝对路径，都认。"""
    text = str(raw or "").strip().strip('"').strip("'")
    if not text:
        raise RuntimeError("「视频文件」没填。选一个视频，或填 input 目录下的相对路径。")
    if os.path.isfile(text):
        return os.path.abspath(text)

    candidates = []
    if text.startswith("/"):
        candidates.append(os.path.join(folder_paths.base_path, text.lstrip("/\\")))
    else:
        candidates.append(os.path.join(folder_paths.get_input_directory(), text))
        candidates.append(os.path.join(folder_paths.base_path, "fxai", "video", text))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise RuntimeError(f"找不到视频文件：{text}")


def _parse_rate(text):
    try:
        num, den = str(text).split("/")
        num, den = float(num), float(den)
        return num / den if den else 0.0
    except Exception:
        return 0.0


def probe_video(video_path):
    cmd = [
        'ffprobe', video_path,
        '-hide_banner', '-loglevel', 'error',
        '-show_streams', '-show_format',
        '-of', 'json',
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, shell=os.name == 'nt')
    if not result.stdout:
        raise RuntimeError(f"ffprobe 读不到这个视频：{video_path}\n{result.stderr.strip()}")

    info = json.loads(result.stdout)
    stream = next((s for s in info.get('streams', []) if s.get('codec_type') == 'video'), None)
    if stream is None:
        raise RuntimeError(f"这个文件里没有视频流：{video_path}")

    fps = _parse_rate(stream.get('r_frame_rate')) or _parse_rate(stream.get('avg_frame_rate')) or 30.0
    duration = float(stream.get('duration') or 0.0) or float(info.get('format', {}).get('duration') or 0.0)
    frames = int(stream.get('nb_frames') or 0) or round_half_up(duration * fps)
    if frames <= 0:
        raise RuntimeError(f"读不出这个视频的帧数：{video_path}")

    return {
        "path": video_path,
        "fps": round(fps, 6),
        "width": int(stream.get('width') or 0),
        "height": int(stream.get('height') or 0),
        "duration": round(duration, 6),
        "frames": frames,
    }


def _start_seconds(start_frame, fps):
    """帧号 → ffmpeg 定位秒数。

    -ss 的判据是「这个时间点落在哪一帧的时间区间里」（左闭右开），
    所以时间点要略大于帧起点：正好等于帧起点时，浮点误差往下一偏就
    退回上一帧，两段就会共用同一张尾帧/首帧。
    加 0.1ms 足够压住误差，又远小于一帧（24fps 是 41.7ms），不会跨帧。
    """
    return max(0.0, int(start_frame) / fps + 0.0001)


def decode_frame_range(video_path, start_frame, frame_count, fps, width, height):
    cmd = [
        'ffmpeg',
        '-ss', f'{_start_seconds(start_frame, fps):.6f}',
        '-i', video_path,
        '-frames:v', str(max(1, int(frame_count))),
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-vsync', '0',
        '-hide_banner', '-loglevel', 'error',
        '-',
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               shell=os.name == 'nt')
    raw, err = process.communicate()
    if process.returncode != 0 and not raw:
        raise RuntimeError(f"ffmpeg 解码失败：{err.decode('utf-8', 'ignore').strip()}")

    frame_size = width * height * 3
    count = len(raw) // frame_size
    if count <= 0:
        raise RuntimeError(f"这一段没解到帧：start={start_frame} frames={frame_count}")
    buffer = np.frombuffer(raw[:count * frame_size], dtype=np.uint8)
    frames = buffer.reshape(count, height, width, 3).astype(np.float32) / 255.0
    return torch.from_numpy(frames)


def decode_audio_range(video_path, start_frame, frame_count, fps):
    """抽同一段的音轨；没有音轨返回 None。

    起止和画面用同一套换算，两边的段长才会一致，配音不会错位。
    """
    duration = max(1e-6, int(frame_count) / fps)
    cmd = [
        'ffmpeg',
        '-ss', f'{_start_seconds(start_frame, fps):.6f}',
        '-i', video_path,
        '-t', f'{duration:.6f}',
        '-vn',
        '-f', 'f32le',
        '-ar', str(AUDIO_SAMPLE_RATE),
        '-ac', '2',
        '-hide_banner', '-loglevel', 'error',
        '-',
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               shell=os.name == 'nt')
    raw, _ = process.communicate()
    if not raw:
        print(f"[凤希AI] 视频没有音轨，音频输出为空：{video_path}")
        return None

    samples = np.frombuffer(raw, dtype=np.float32)
    samples = samples[:samples.size - samples.size % 2].reshape(-1, 2)
    waveform = torch.from_numpy(np.ascontiguousarray(samples.T)).unsqueeze(0).float()
    return {"waveform": waveform, "sample_rate": AUDIO_SAMPLE_RATE}


def format_time(seconds):
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    return f"{minutes}:{seconds - minutes * 60:06.3f}"


def format_fps(fps):
    """帧率显示：整数就不带小数点，23.976 这种才保留三位。

    素材基本都是 24/25/30，写成 24.000 看着像有小数；但真遇到
    NTSC 的 23.976 又不能砍成 24，所以只在非整数时才带小数。
    """
    value = float(fps)
    return f"{value:.0f}" if abs(value - round(value)) < 1e-6 else f"{value:.3f}"


def filmstrip_path(video_path, tiles):
    """胶片带缓存位置。同一视频 + 同一分格数共用一个文件，mtime 变了就重抽。"""
    cache_dir = os.path.join(folder_paths.base_path, "fxai", "video", ".thumbs")
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5(os.path.normcase(video_path).encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"strip_{key}_{int(tiles)}.jpg")


def make_filmstrip(video_path, out_path, duration, tiles=40, tile_width=160, tile_height=90):
    """整条视频等间隔抽 tiles 帧，横向拼成一张图，给面板当时间轴底图。

    统一按 160x90 缩放再补边，竖屏也不会把画布撑高。
    """
    tiles = max(4, min(int(tiles), 80))
    interval = max(0.02, float(duration) / tiles)
    filters = (
        f"fps=1/{interval:.6f},"
        f"scale={tile_width}:{tile_height}:force_original_aspect_ratio=decrease,"
        f"pad={tile_width}:{tile_height}:(ow-iw)/2:(oh-ih)/2,"
        f"tile={tiles}x1"
    )
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vf', filters,
        '-frames:v', '1', '-q:v', '4',
        '-hide_banner', '-loglevel', 'error',
        out_path,
    ]
    try:
        subprocess.run(cmd, timeout=180, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return False
    return os.path.isfile(out_path) and os.path.getsize(out_path) > 0
