import json
import math
import mimetypes
import os
import wave
import torch
import folder_paths
import numpy as np
import server
import subprocess
import time
from aiohttp import web
# 导入工具类全局音频后缀常量
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

def list_input_audio_files():
    input_dir = folder_paths.get_input_directory()
    if not input_dir or not os.path.isdir(input_dir):
        return []
    # 直接使用utils统一后缀，不再本地定义
    discovered = []
    for root, _dirs, files in os.walk(input_dir):
        for filename in files:
            # 统一endswith判断方式，和load节点保持一致
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
    def select_segment(self, 音频文件="", 关键帧JSON="[]", 跳过初始段=False, 包含尾部段=True, 是否平均分段=True, 平均分段时长=15, 音频=None):
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
        return (selected, segment_list)
		

    @classmethod
    def IS_CHANGED(s):
        return str(time.time())

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