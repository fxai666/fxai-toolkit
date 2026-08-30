import json
import math
import mimetypes
import os
import torch
import folder_paths
import numpy as np
import server
import subprocess
import time
from aiohttp import web

from fxai_audio_utils import (
    resolve_audio_path,
    load_audio_tensor_from_file,
    normalize_audio_tensor,
    slice_audio,
    read_waveform_peaks,
    get_wav_path,
    AUDIO_EXTENSIONS
)

MAX_MARKERS = 64
MIN_SEGMENT_DUR = 0.1
_H3_FPS = 24


def align_down_h3(frames):
    # 最大的 17k+5 <= frames（不超目标时长）
    if frames < 5:
        return 5
    return 17 * ((frames - 5) // 17) + 5


def align_up_h3(frames):
    # 最小的 17k+5 >= frames
    if frames <= 5:
        return 5
    k = (frames - 5 + 16) // 17
    return 17 * k + 5


def align_frames_last_h3(segment_seconds, avg_dur=0.0):
    """每段先按 17k+5 向下取整算出实际帧数，最后一段 = 总帧数 - 前面各段帧数和，
    再向上对齐到最近的 17k+5（保证 H3 支持该帧数，总时长略增）；
    末段结果小于 3s（avg_dur>10s 按 5s）时并入其前一段。

    前面各段对齐后舍掉的帧计入末段，末段向上对齐补齐 H3 合法性。
    """
    total = round(sum(segment_seconds) * _H3_FPS)
    frames = [align_down_h3(round(s * _H3_FPS)) for s in segment_seconds]
    frames[-1] = align_up_h3(total - sum(frames[:-1]))
    min_threshold = 5.0 if avg_dur > 10.0 else 3.0
    if len(frames) >= 2 and frames[-1] < min_threshold * _H3_FPS:
        frames[-2] += frames[-1]
        frames.pop()
    return frames

def list_input_audio_files():
    input_dir = folder_paths.get_input_directory()
    if not input_dir or not os.path.isdir(input_dir):
        return []
    discovered = []
    for root, _dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(AUDIO_EXTENSIONS):
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, input_dir).replace("\\", "/")
                discovered.append(rel_path)
    return sorted(discovered)

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

def normalize_keyframe_list(keyframes, total_duration=None):
    seen = set()
    norm = []
    for sec in keyframes or []:
        sec = max(0.0, float(sec))
        if total_duration:
            sec = min(sec, total_duration - 0.001)
        bucket = round(sec * 1000)
        if bucket not in seen:
            seen.add(bucket)
            norm.append(sec)
    norm.sort()
    return norm[:MAX_MARKERS]

def build_segments(total_duration, keyframes, skip_initial, include_tail, is_average_split=False, avg_dur=0.0):
    total_duration = max(0.0, total_duration)
    markers = normalize_keyframe_list(keyframes, total_duration)
    segments = []
    if not markers:
        segments = [(0.0, total_duration)]
    else:
        points = [0.0] + markers + [total_duration]
        for i in range(len(points)-1):
            s, e = points[i], points[i+1]
            if e > s:
                segments.append((s, e))
    if skip_initial and len(segments) > 0:
        segments = segments[1:]
    if not include_tail and len(segments) > 0:
        segments = segments[:-1]
    clean_segs = []
    for s, e in segments:
        if e - s >= MIN_SEGMENT_DUR:
            clean_segs.append((s, e))
    segments = clean_segs
    if not segments:
        segments = [(0.0, total_duration)]
    if is_average_split and avg_dur > 0:
        s_total = segments[0][0]
        e_total = segments[-1][1]
        new_segs = []
        curr = s_total
        while curr < e_total:
            end = curr + avg_dur
            if end > e_total:
                end = e_total
            new_segs.append((curr, end))
            curr = end
        segments = new_segs

        # 最后一段太短就并入前一段：>10秒分段按5秒阈值，否则按3秒
        min_threshold = 5.0 if avg_dur > 10.0 else 3.0
        if len(segments) >= 2:
            last_start, last_end = segments[-1]
            last_length = last_end - last_start
            if last_length <= min_threshold:
                segments.pop()
                prev_start, _ = segments[-1]
                segments[-1] = (prev_start, last_end)

    return segments, sum(e-s for s, e in segments)

class FxAiAudioSegmenterV2:
    @classmethod
    def INPUT_TYPES(cls):
        files = list_input_audio_files() or [""]
        return {
            "required": {
                "音频文件": (files, {"default": files[0]}),
                "关键帧JSON": ("STRING", {"default": "[]", "multiline": False}),
                "跳过初始段": ("BOOLEAN", {"default": False}),
                "包含尾部段": ("BOOLEAN", {"default": True}),
                "是否平均分段": ("BOOLEAN", {"default": True}),
                "平均分段时长": ("FLOAT", {"default": 15.00, "step": 0.01, "round": 0.01}),
                "目标模型": (["LTX", "MiniMax"], {"default": "LTX",
                    "tooltip": "LTX 按原始时长输出；MiniMax 把每段时长就近对齐到 17k+5 帧网格，避免逐段舍帧在长循环中累积误差"}),
            },
            "optional": {
                "音频": ("AUDIO", {"forceInput": True}),
            }
        }
    RETURN_TYPES = ("AUDIO", "LIST")
    RETURN_NAMES = ("音频", "分段列表")
    FUNCTION = "select_segment"
    CATEGORY = "凤希AI/音频"
    @classmethod
    def VALIDATE_INPUTS(cls, 音频文件="", 关键帧JSON="[]", **_kwargs):
        try:
            if 音频文件:
                resolve_audio_path(音频文件)
            parse_keyframe_list(关键帧JSON)
            return True
        except Exception as e:
            return str(e)
    def select_segment(self, 音频文件="", 目标模型="LTX", 关键帧JSON="[]", 跳过初始段=False, 包含尾部段=True, 是否平均分段=True, 平均分段时长=15, 音频=None):
        target_model = str(目标模型).lower()
        audio = 音频 or load_audio_tensor_from_file(音频文件)
        waveform, sample_rate = normalize_audio_tensor(audio)
        total_duration = waveform.shape[-1] / sample_rate if sample_rate else 0.0
        keyframes = parse_keyframe_list(关键帧JSON)
        segments, _ = build_segments(total_duration, keyframes, 跳过初始段, 包含尾部段, 是否平均分段, 平均分段时长)
        start_sec = segments[0][0]
        end_sec = segments[-1][1]
        start_frame = int(start_sec * sample_rate)
        end_frame = int(end_sec * sample_rate)
        selected = slice_audio(audio, start_frame, end_frame)
        segment_list = [round(e - s, 2) for s, e in segments]
        if target_model == "minimax":
            # MiniMax：在已选择的区间内按帧数重新分段
            interval_duration = end_sec - start_sec
            total_frames = round(interval_duration * _H3_FPS)
            frames_per_seg = align_down_h3(round(平均分段时长 * _H3_FPS)) if 平均分段时长 > 0 else 226
            num_full_segs = total_frames // frames_per_seg
            tail_frames = total_frames - num_full_segs * frames_per_seg
            # 尾段太短合并到前一段（与 build_segments 公共阈值一致）
            min_threshold_frames = int((5.0 if 平均分段时长 > 10.0 else 3.0) * _H3_FPS)
            if tail_frames > 0 and tail_frames < min_threshold_frames and num_full_segs > 0:
                num_full_segs -= 1
                tail_frames += frames_per_seg
            frames = [frames_per_seg] * num_full_segs
            if tail_frames > 0:
                frames.append(tail_frames)
            # 帧数→秒数（5位小数）
            segment_list = [round(f / _H3_FPS, 5) for f in frames]
        return (selected, segment_list)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

async def simple_audio_file(request):
    audio_file = request.query.get("audio_file", "")
    try:
        path = resolve_audio_path(audio_file)
        return web.FileResponse(path, headers={"Content-Type": "audio/wav"})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)

async def simple_audio_waveform(request):
    audio_file = request.query.get("audio_file", "")
    bins = request.query.get("bins", "1400")
    try:
        data = read_waveform_peaks(audio_file, bins=int(bins))
        wav_file = get_wav_path(audio_file)
        data["audio_url"] = f"/fxai/audio/segments/file?audio_file={wav_file}"
        return web.json_response(data)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)

prompt_server_instance = getattr(server.PromptServer, "instance", None)
if prompt_server_instance:
    prompt_server_instance.routes.get("/fxai/audio/segments/file")(simple_audio_file)
    prompt_server_instance.routes.get("/fxai/audio/segments/waveform")(simple_audio_waveform)