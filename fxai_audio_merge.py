# ===================== 所有导入统一放在顶部 =====================
import os
import re
import torch
import subprocess
import tempfile
import wave
import numpy as np

# ===================== 工具函数 =====================
def get_empty_audio(sr=44100, channels=2):
    waveform = torch.zeros((1, channels, 1), dtype=torch.float32)
    return {"waveform": waveform, "sample_rate": sr}

def list_audios(target_dir):
    if not os.path.isdir(target_dir):
        return []
    pattern = re.compile(r'(.+)\.(mp3|wav|ogg|flac|m4a)$', re.IGNORECASE)
    files = []
    for f in sorted(os.listdir(target_dir)):
        fp = os.path.join(target_dir, f)
        if os.path.isfile(fp) and pattern.match(f):
            files.append(f)
    return files

# ===================== 纯Python音频加载（无组件依赖） =====================
def _load_audio_tensor_from_file(audio_file_path):
    audio_path = audio_file_path
    ext = os.path.splitext(audio_path)[1].lower()

    with wave.open(audio_path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        sr = wav_file.getframerate()
        frames = wav_file.getnframes()
        data = wav_file.readframes(frames)

    if frames <= 0 or sr <= 0:
        raise ValueError("无效的WAV文件")

    if sampwidth == 1:
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
        arr = (arr - 128.0) / 128.0
    elif sampwidth == 2:
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        arr = arr / 32768.0
    elif sampwidth == 3:
        raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
        signed = (raw[:, 0].astype(np.int32) |
                  (raw[:, 1].astype(np.int32) << 8) |
                  (raw[:, 2].astype(np.int32) << 16))
        sign_mask = 1 << 23
        signed = (signed ^ sign_mask) - sign_mask
        arr = signed.astype(np.float32) / float(1 << 23)
    elif sampwidth == 4:
        arr = np.frombuffer(data, dtype=np.int32).astype(np.float32)
        arr = arr / float(1 << 31)
    else:
        raise ValueError(f"不支持的采样位宽: {sampwidth}")

    if channels > 1:
        arr = arr.reshape(-1, channels).T
    else:
        arr = arr.ravel()[None, :]

    waveform = torch.from_numpy(arr).unsqueeze(0).float()
    return {"waveform": waveform, "sample_rate": sr}

# ===================== 音频合并主类 =====================
class FxAiAudioMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文件夹路径": ("STRING", {"default": ""}),
                "目标采样率": ("INT", {"default": 44100, "min": 8000, "max": 96000}),
                "目标声道": ("INT", {"default": 2, "min": 1, "max": 2}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("合并音频",)
    FUNCTION = "merge_audios"
    CATEGORY = "凤希AI/音频"

    def merge_audios(self, 文件夹路径="", 目标采样率=44100, 目标声道=2):
        try:
            if not 文件夹路径 or not os.path.isdir(文件夹路径):
                print("❌ 音频合并：目录不存在")
                return (get_empty_audio(目标采样率, 目标声道),)

            audio_files = list_audios(文件夹路径)
            if not audio_files:
                print("❌ 音频合并：目录无音频文件")
                return (get_empty_audio(目标采样率, 目标声道),)

            with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".txt") as f:
                for fname in audio_files:
                    fpath = os.path.join(文件夹路径, fname)
                    f.write(f"file '{fpath.replace("'", "'\\''")}'\n")
                file_list_path = f.name

            temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", file_list_path,
                "-ar", str(目标采样率),
                "-ac", str(目标声道),
                "-c:a", "pcm_s16le",
                "-y", temp_out
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8"
            )

            os.unlink(file_list_path)

            if result.returncode != 0 or not os.path.exists(temp_out):
                print(f"❌ FFmpeg 合并失败：{result.stderr}")
                if os.path.exists(temp_out):
                    os.unlink(temp_out)
                return (get_empty_audio(目标采样率, 目标声道),)

            # 纯Python加载合并后的音频
            audio_out = _load_audio_tensor_from_file(temp_out)
            os.unlink(temp_out)

            print(f"✅ 合并完成：{len(audio_files)} 个音频 | 采样率:{audio_out['sample_rate']}Hz | {目标声道}声道")
            return (audio_out,)

        except Exception as e:
            print(f"❌ 合并异常：{str(e)}")
            return (get_empty_audio(目标采样率, 目标声道),)