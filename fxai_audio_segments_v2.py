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
from aiohttp import web

MAX_MARKERS = 64


def _strip_path(path):
    path = (path or "").strip()
    if path.startswith('"'):
        path = path[1:]
    if path.endswith('"'):
        path = path[:-1]
    return path


def _get_wav_path(file_path):
    base, _ext = os.path.splitext(file_path)
    return base + ".wav"


def _list_input_audio_files():
    input_dir = folder_paths.get_input_directory()
    if not input_dir or not os.path.isdir(input_dir):
        return []
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm", ".wma", ".ac3"}
    discovered = []
    for root, _dirs, files in os.walk(input_dir):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in audio_extensions:
                continue
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, input_dir).replace("\\", "/")
            discovered.append(rel_path)
    return sorted(discovered)


def _resolve_audio_path(audio_file):
    audio_file = _strip_path(audio_file)
    if not audio_file:
        raise ValueError("音频文件路径为空")

    wav_file = _get_wav_path(audio_file)
    wav_full_path = os.path.join(folder_paths.get_input_directory(), wav_file)
    # wav已存在直接返回
    if os.path.exists(wav_full_path):
        return wav_full_path
    # wav不存在则找原格式文件转码
    original_full_path = os.path.join(folder_paths.get_input_directory(), audio_file)
    if not os.path.exists(original_full_path):
        raise ValueError(f"未找到音频文件: {audio_file}")

    cmd = [
        "ffmpeg", "-i", original_full_path,
        "-ac", "1", "-f", "wav", "-y", wav_full_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    # 转码成功删除原文件
    if os.path.exists(wav_full_path):
        try:
            os.remove(original_full_path)
        except Exception:
            pass
    return wav_full_path


def _read_wav(wav_path):
    with wave.open(wav_path, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.getnframes()
        data = wf.readframes(frames)

    if sampwidth == 1:
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
        arr = (arr - 128) / 128.0
    elif sampwidth == 2:
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
        signed = (raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16))
        sign_mask = 1 << 23
        signed = (signed ^ sign_mask) - sign_mask
        arr = signed.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        arr = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError("不支持的位宽")

    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)
    return arr.ravel(), sr


def _load_audio_tensor_from_file(audio_file):
    audio_path = _resolve_audio_path(audio_file)
    arr, sr = _read_wav(audio_path)
    waveform = torch.from_numpy(arr).unsqueeze(0).float()
    return {"waveform": waveform, "sample_rate": sr}


def _read_waveform_peaks(audio_file, bins=1400):
    audio = _load_audio_tensor_from_file(audio_file)
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    waveform_np = waveform.numpy()
    if waveform_np.shape[1] > 1:
        samples = np.mean(np.abs(waveform_np), axis=1)[0]
    else:
        samples = np.abs(waveform_np[0, 0])
    frame_count = len(samples)
    bins = max(64, min(int(bins), 4096))
    peaks = []
    if samples.size > 0:
        edges = np.linspace(0, samples.size, num=bins + 1, dtype=np.int64)
        for idx in range(bins):
            start = edges[idx]
            end = edges[idx + 1]
            peaks.append(float(np.max(samples[start:end])) if end > start else 0.0)
    duration = frame_count / sample_rate if sample_rate > 0 else 0.0
    return {
        "duration": duration,
        "sample_rate": sample_rate,
        "peaks": peaks,
        "audio_path": _resolve_audio_path(audio_file),
    }


def _parse_keyframe_list(value):
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


def _normalize_keyframe_list(keyframes, total_duration=None):
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


def _normalize_audio_tensor(audio):
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    return waveform, sample_rate


def _slice_audio(audio, start_frame, end_frame):
    waveform, sample_rate = _normalize_audio_tensor(audio)
    start_frame = max(0, int(start_frame))
    end_frame = max(start_frame + 1, int(end_frame))
    return {"waveform": waveform[..., start_frame:end_frame], "sample_rate": sample_rate}


def _build_segments(total_duration, keyframes, skip_initial, include_tail, is_average_split=False, avg_dur=0.0):
    total_duration = max(0.0, total_duration)
    markers = _normalize_keyframe_list(keyframes, total_duration)
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
    if segments and (segments[-1][1] - segments[-1][0] < 0.1):
        segments.pop()
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
        files = _list_input_audio_files() or [""]
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
                "刷新标记": ("INT", {"forceInput": True}),
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
                _resolve_audio_path(音频文件)
            _parse_keyframe_list(关键帧JSON)
            return True
        except Exception as e:
            return str(e)

    def select_segment(self, 音频文件="", 关键帧JSON="[]", 跳过初始段=False, 包含尾部段=True, 是否平均分段=True, 平均分段时长=15, 刷新标记=0, 音频=None):
        audio = 音频 or _load_audio_tensor_from_file(音频文件)
        waveform, sample_rate = _normalize_audio_tensor(audio)
        total_duration = waveform.shape[-1] / sample_rate if sample_rate else 0.0
        keyframes = _parse_keyframe_list(关键帧JSON)
        segments, _ = _build_segments(total_duration, keyframes, 跳过初始段, 包含尾部段, 是否平均分段, 平均分段时长)
        start_sec = segments[0][0]
        end_sec = segments[-1][1]
        start_frame = int(start_sec * sample_rate)
        end_frame = int(end_sec * sample_rate)
        selected = _slice_audio(audio, start_frame, end_frame)
        segment_list = [round(e - s, 2) for s, e in segments]
        return (selected, segment_list)


async def simple_audio_file(request):
    audio_file = request.query.get("audio_file", "")
    try:
        path = _resolve_audio_path(audio_file)
        return web.FileResponse(path, headers={"Content-Type": "audio/wav"})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)


async def simple_audio_waveform(request):
    audio_file = request.query.get("audio_file", "")
    bins = request.query.get("bins", "1400")
    try:
        data = _read_waveform_peaks(audio_file, bins=int(bins))
        wav_file = _get_wav_path(audio_file)
        data["audio_url"] = f"/fxai/audio/segments/file?audio_file={wav_file}"
        return web.json_response(data)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)


_prompt_server_instance = getattr(server.PromptServer, "instance", None)
if _prompt_server_instance:
    _prompt_server_instance.routes.get("/fxai/audio/segments/file")(simple_audio_file)
    _prompt_server_instance.routes.get("/fxai/audio/segments/waveform")(simple_audio_waveform)