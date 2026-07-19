import json
import math
import mimetypes
import os
import torch
import folder_paths
import numpy as np
import server
import subprocess
import soundfile as sf
from aiohttp import web

AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', ".ac3")

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

    if audio_file.startswith("/"):
        base_dir = folder_paths.base_path
        target_relative = audio_file.lstrip("/")
    else:
        base_dir = folder_paths.get_input_directory()
        target_relative = audio_file

    wav_file = get_wav_path(target_relative)
    wav_full_path = os.path.join(base_dir, wav_file)
    if os.path.exists(wav_full_path):
        return wav_full_path
		
    print(f"{wav_full_path}")
    original_full_path = os.path.join(base_dir, target_relative)
    if not os.path.exists(original_full_path):
        raise ValueError(f"未找到音频文件: {audio_file}")

    cmd = [
        "ffmpeg", "-i", original_full_path,
        "-f", "wav", "-y", wav_full_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)

    if os.path.exists(wav_full_path):
        try:
            os.remove(original_full_path)
        except Exception:
            pass
    return wav_full_path

def _to_stereo(arr):
    if arr.ndim == 1:
        return np.stack([arr, arr], axis=0)
    if arr.shape[1] == 1:
        return np.repeat(arr, 2, axis=1).T
    return arr[:, :2].T

def load_audio_tensor_from_file(audio_file):
    audio_path = resolve_audio_path(audio_file)
    arr, sr = sf.read(audio_path)
    arr = arr.astype(np.float32)
    arr = _to_stereo(arr)
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

def load_wav_tensor(wav_path):
    arr, sr = sf.read(wav_path)
    arr = arr.astype(np.float32)
    arr = _to_stereo(arr)
    waveform = torch.from_numpy(arr).float().unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sr}

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