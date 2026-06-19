import os
import torch
from fxai_audio_utils import (
    AUDIO_EXTENSIONS,
    load_audio_tensor_from_file,
    normalize_audio_tensor,
    slice_audio
)

class FxAiLoadAudioByIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "音频文件夹路径": ("STRING", {"multiline": False}),
                "音频索引": ("INT", {"default": 0, "min": 0}),
            },
            "optional": {
                "刷新标记": ("INT", {"forceInput": True}),
                "起始秒数": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.001, "precision": 3}),
                "截取时长秒数": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.001, "precision": 3}),
            }
        }
    RETURN_TYPES = ("AUDIO", "INT", "STRING", "INT")
    RETURN_NAMES = ("音频", "采样率", "当前音频路径", "总音频数量")
    FUNCTION = "load_audio"
    CATEGORY = "凤希AI/音频"

    def load_audio(self, 音频文件夹路径, 音频索引, 刷新标记=0, 起始秒数=0.0, 截取时长秒数=0.0):
        folder_path = 音频文件夹路径.strip()

        if not os.path.isdir(folder_path):
            raise RuntimeError(f"文件夹不存在：{folder_path}")

        audio_files = []
        # 换回你原来的判断方式，简单稳定，不会报元组错误
        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(AUDIO_EXTENSIONS):
                full_path = os.path.join(folder_path, filename)
                audio_files.append(full_path)

        total_audios = len(audio_files)

        if total_audios == 0 or 音频索引 >= total_audios:
            empty_waveform = torch.zeros(1, 2, 44100, dtype=torch.float32)
            empty_audio = {
                "waveform": empty_waveform,
                "sample_rate": 44100
            }
            return (empty_audio, 44100, "索引越界-无音频", total_audios)

        target_path = audio_files[音频索引 % total_audios]

        audio_dict = load_audio_tensor_from_file(target_path)
        waveform, sample_rate = normalize_audio_tensor(audio_dict)

        total_sec = waveform.shape[-1] / sample_rate
        start_sec = max(0.0, 起始秒数)
        end_sec = total_sec
        if 截取时长秒数 > 0:
            end_sec = start_sec + 截取时长秒数
        end_sec = min(end_sec, total_sec)

        start_frame = int(start_sec * sample_rate)
        end_frame = int(end_sec * sample_rate)
        final_audio = slice_audio(audio_dict, start_frame, end_frame)

        # 复制双声道，兼容旧工作流
        final_audio["waveform"] = final_audio["waveform"].repeat(1, 2, 1)

        return (final_audio, sample_rate, target_path, total_audios)