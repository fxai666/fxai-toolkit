import os
import re
import torch
import subprocess
import tempfile
import torchaudio

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
            # 1. 检查目录
            if not 文件夹路径 or not os.path.isdir(文件夹路径):
                print("❌ 音频合并：目录不存在")
                return (get_empty_audio(目标采样率, 目标声道),)

            # 2. 获取已排序的音频文件
            audio_files = list_audios(文件夹路径)
            if not audio_files:
                print("❌ 音频合并：目录无音频文件")
                return (get_empty_audio(目标采样率, 目标声道),)

            # 3. 创建 FFmpeg 列表文件（必须用这个才能无缝合并）
            with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".txt") as f:
                for fname in audio_files:
                    fpath = os.path.join(文件夹路径, fname)
                    # FFmpeg 格式：file '文件路径'
                    f.write(f"file '{fpath.replace("'", "'\\''")}'\n")
                file_list_path = f.name

            # 4. 创建临时输出文件
            temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

            # ======================
            # FFmpeg 核心合并命令
            # ======================
            cmd = [
                "ffmpeg",
                "-f", "concat",        # 合并模式
                "-safe", "0",          # 允许任意路径
                "-i", file_list_path,  # 输入列表
                "-ar", str(目标采样率), # 目标采样率
                "-ac", str(目标声道),   # 目标声道
                "-c:a", "pcm_s16le",   # 标准WAV格式
                "-y", temp_out         # 覆盖输出
            ]

            # 执行命令
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8"
            )

            # 清理临时列表文件
            os.unlink(file_list_path)

            # 检查合并是否成功
            if result.returncode != 0 or not os.path.exists(temp_out):
                print(f"❌ FFmpeg 合并失败：{result.stderr}")
                os.unlink(temp_out)
                return (get_empty_audio(目标采样率, 目标声道),)

            # 5. 加载合并好的音频给 ComfyUI
            waveform, sr = torchaudio.load(temp_out)
            os.unlink(temp_out)  # 用完就删

            # 包装成 ComfyUI 格式
            waveform = waveform.unsqueeze(0).float()
            audio_out = {
                "waveform": waveform,
                "sample_rate": sr
            }

            print(f"✅ FFmpeg 合并完成：{len(audio_files)} 个音频 | 采样率:{sr}Hz | {目标声道}声道")
            return (audio_out,)

        except Exception as e:
            print(f"❌ 合并异常：{str(e)}")
            return (get_empty_audio(目标采样率, 目标声道),)