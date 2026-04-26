import os
import torch
import torchaudio
import numpy as np

# 工具函数：加载单段音频（严格对齐ComfyUI音频张量标准）
def load_single_audio(audio_path, start_seconds=0.0, duration_seconds=0.0):
    try:
        # 加载音频
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 维度修复
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() > 2:
            waveform = waveform.squeeze()[:2, :]
        
        # ===================== 按你的思路：取最小值 =====================
        total_samples = waveform.shape[-1]
        
        # 起始采样点（安全限制在 0 ~ 总长度之间）
        start_sample = int(start_seconds * sample_rate)
        start_sample = max(0, min(start_sample, total_samples))
        
        # 剩余长度 = 总长度 - 起始位置
        remaining_samples = total_samples - start_sample

        # 用户想要截取的长度
        if duration_seconds > 0:
            desired_samples = int(duration_seconds * sample_rate)
        else:
            desired_samples = remaining_samples  # 时长=0 → 取全部剩余

        # 最终长度 = 取小的那个（你的核心思路！）
        final_samples = min(desired_samples, remaining_samples)

        # 安全截取
        waveform = waveform[..., start_sample : start_sample + final_samples]
        # =================================================================

        # 空音频检测
        if waveform.size(-1) == 0:
            raise RuntimeError("截取后无有效音频")

        # 转为ComfyUI标准格式
        audio_tensor = waveform.to(torch.float32).unsqueeze(0)
        return audio_tensor, sample_rate
    except Exception as e:
        raise RuntimeError(f"加载/截取音频失败 {audio_path}：{str(e)}")

# 支持的音频格式
AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma')

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
        for f in os.listdir(folder_path):
            if f.lower().endswith(AUDIO_EXTENSIONS):
                full_path = os.path.join(folder_path, f)
                audio_files.append(full_path)
        
        audio_files.sort()
        total_audios = len(audio_files)
        
        if total_audios == 0:
            raise RuntimeError(f"文件夹中没有找到音频：{folder_path}")
        if 音频索引 >= total_audios:
            raise RuntimeError(f"索引越界！共 {total_audios} 段音频，索引范围：0 ~ {total_audios-1}")
        
        target_path = audio_files[音频索引]
        if not os.path.exists(target_path):
            raise RuntimeError(f"音频文件不存在：{target_path}")
        if os.path.getsize(target_path) < 100:
            raise RuntimeError(f"音频文件过小/为空：{target_path}")
        
        audio_tensor, sample_rate = load_single_audio(
            target_path,
            start_seconds=起始秒数,
            duration_seconds=截取时长秒数
        )
        
        audio_output = {
            "waveform": audio_tensor,
            "sample_rate": sample_rate
        }
        
        return (audio_output, sample_rate, target_path, total_audios)