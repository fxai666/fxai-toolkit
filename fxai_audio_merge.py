import os
import re
import torch
import torchaudio
from torchaudio.transforms import Resample

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
        if not os.path.isfile(fp):
            continue
        m = pattern.match(f)
        if m:
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
            if not 文件夹路径 or not os.path.isdir(文件夹路径):
                print("❌ 音频合并：目录不存在")
                return (get_empty_audio(目标采样率, 目标声道),)

            # ======================
            # 直接用你原来的函数！已经排序好了！
            # ======================
            audio_files = list_audios(文件夹路径)

            if not audio_files:
                print("❌ 音频合并：目录无音频文件")
                return (get_empty_audio(目标采样率, 目标声道),)

            all_wavs = []
            device = "cuda" if torch.cuda.is_available() else "cpu"

            for filename in audio_files:
                try:
                    fp = os.path.join(文件夹路径, filename)
                    wav, sr = torchaudio.load(fp)

                    # 重采样
                    if sr != 目标采样率:
                        resampler = Resample(sr, 目标采样率).to(device)
                        wav = resampler(wav.to(device)).cpu()

                    # 统一声道
                    c = wav.shape[0]
                    if c < 目标声道:
                        wav = wav.repeat(目标声道, 1)
                    elif c > 目标声道:
                        wav = wav[:目标声道]

                    all_wavs.append(wav)
                except Exception as e:
                    print(f"⚠️ 跳过文件 {filename}: {str(e)}")

            if not all_wavs:
                return (get_empty_audio(目标采样率, 目标声道),)

            combined = torch.cat(all_wavs, dim=-1)
            waveform = combined.unsqueeze(0).float()

            audio_out = {
                "waveform": waveform,
                "sample_rate": 目标采样率
            }

            print(f"✅ 合并完成：{len(audio_files)} 个音频 | 采样率:{目标采样率}Hz | {目标声道}声道")
            return (audio_out,)

        except Exception as e:
            print(f"❌ 合并异常：{str(e)}")
            return (get_empty_audio(),)