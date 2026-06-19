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

AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma')

def strip_path(path):
    path = (path or "").strip()
    if path.startswith('"'):
        path = path[1:]
    if path.endswith('"'):
        path = path[:-1]
    return path

def get_wav_path(file_path):
    base, _ext = os.path.splitext(file_path)
    return base + ".wav"

def resolve_audio_path(audio_file):
    audio_file = strip_path(audio_file)
    if not audio_file:
        raise ValueError("音频文件路径为空")
    wav_file = get_wav_path(audio_file)
    wav_full_path = os.path.join(folder_paths.get_input_directory(), wav_file)
    if os.path.exists(wav_full_path):
        return wav_full_path
    original_full_path = os.path.join(folder_paths.get_input_directory(), audio_file)
    if not os.path.exists(original_full_path):
        raise ValueError(f"未找到音频文件: {audio_file}")
    cmd = [
        "ffmpeg", "-i", original_full_path,
        "-ac", "1", "-f", "wav", "-y", wav_full_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    if os.path.exists(wav_full_path):
        try:
            os.remove(original_full_path)
        except Exception:
            pass
    return wav_full_path

def read_wav(wav_path):
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

def load_audio_tensor_from_file(audio_file):
    audio_path = resolve_audio_path(audio_file)
    arr, sr = read_wav(audio_path)
    waveform = torch.from_numpy(arr).unsqueeze(0).float()
    return {"waveform": waveform, "sample_rate": sr}

def read_waveform_peaks(audio_file, bins=1400):
    audio = load_audio_tensor_from_file(audio_file)
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
        "audio_path": resolve_audio_path(audio_file),
    }

def normalize_audio_tensor(audio):
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    return waveform, sample_rate

def slice_audio(audio, start_frame, end_frame):
    waveform, sample_rate = normalize_audio_tensor(audio)
    start_frame = max(0, int(start_frame))
    end_frame = max(start_frame + 1, int(end_frame))
    return {"waveform": waveform[..., start_frame:end_frame], "sample_rate": sample_rate}