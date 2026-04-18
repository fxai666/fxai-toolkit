import os
import torch
import torchaudio
import numpy as np

# 工具函数：加载单段音频（严格对齐ComfyUI音频张量标准）
def load_single_audio(audio_path):
    try:
        # 加载音频：waveform = [channels, samples]（2维），sample_rate = 采样率
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 关键修复1：确保是2维张量（防止单声道/空音频导致维度异常）
        if waveform.dim() == 1:  # 单声道可能被加载为1维，强制转为2维 [1, samples]
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() > 2:  # 异常维度直接截断为前2维
            waveform = waveform.squeeze()[:2, :]  # 最多保留2声道
        
        # 关键修复2：转为 float32 + 严格的 [1, channels, samples] 格式（ComfyUI标准3维）
        # 注意：只扩充1个batch维度，避免多维度叠加
        audio_tensor = waveform.to(torch.float32).unsqueeze(0)
        
        # 额外防护：空音频检测（防止采样点数为0）
        if audio_tensor.size(-1) == 0:
            raise RuntimeError(f"音频文件无有效数据：{audio_path}")
        
        return audio_tensor, sample_rate
    except Exception as e:
        # 增强错误提示，定位具体问题
        raise RuntimeError(f"加载音频失败 {audio_path}：{str(e)}")

# 支持的音频格式
AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma')

class FxAiLoadAudioByIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "音频文件夹路径": ("STRING", {"multiline": False}),
                "音频索引": ("INT", {"default": 0, "min": 0}),
                "刷新标记": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("AUDIO", "INT", "STRING", "INT")
    RETURN_NAMES = ("音频", "采样率", "当前音频路径", "总音频数量")
    FUNCTION = "load_audio"
    CATEGORY = "凤希AI"

    def load_audio(self, 音频文件夹路径, 音频索引, 刷新标记):
        # 1. 清理路径
        folder_path = 音频文件夹路径.strip()
        
        # 2. 检查文件夹是否存在
        if not os.path.isdir(folder_path):
            raise RuntimeError(f"文件夹不存在：{folder_path}")
        
        # 3. 获取文件夹里所有音频（按文件名排序）
        audio_files = []
        for f in os.listdir(folder_path):
            if f.lower().endswith(AUDIO_EXTENSIONS):
                full_path = os.path.join(folder_path, f)
                audio_files.append(full_path)
        
        # 按文件名排序（保证每次顺序一致）
        audio_files.sort()
        
        # 4. 检查是否有音频
        total_audios = len(audio_files)
        if total_audios == 0:
            raise RuntimeError(f"文件夹中没有找到音频：{folder_path}")
        
        # 5. 检查索引是否越界
        if 音频索引 >= total_audios:
            raise RuntimeError(f"索引越界！共 {total_audios} 段音频，索引范围：0 ~ {total_audios-1}")
        
        # 6. 加载选中的音频（新增文件存在性+大小校验）
        target_path = audio_files[音频索引]
        if not os.path.exists(target_path):
            raise RuntimeError(f"音频文件不存在：{target_path}")
        if os.path.getsize(target_path) < 100:  # 过滤空文件/极小无效文件
            raise RuntimeError(f"音频文件过小/为空：{target_path}")
        
        audio_tensor, sample_rate = load_single_audio(target_path)
        
        # 关键修复3：封装为ComfyUI标准的Audio字典（下游节点依赖此格式）
        # 替代直接返回张量，改为返回包含waveform和sample_rate的字典
        audio_output = {
            "waveform": audio_tensor,
            "sample_rate": sample_rate
        }
        
        return (audio_output, sample_rate, target_path, total_audios)