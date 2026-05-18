import datetime
import torch
import math
import torchaudio

class FxAiSingleAudioProc:
    CATEGORY = "凤希AI/音频"
    FUNCTION = "extract_audio_segment"

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("剪切音频", "生成帧数")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "帧率": ("INT", {"default": 24, "min": 1}),
                "当前索引": ("INT", {"default": 0, "min": 0}),
                "帧数对齐基数": ("INT", {"default": 8, "min": 1}),
                "过渡帧数": ("INT", {"default": 1, "min": 0}),

                "统一音量强度": ("INT", {"default": 50, "min": 1, "max": 100}),

                "目标比特率kbps": (["128kbps", "192kbps", "256kbps", "320kbps", "384kbps"], {"default": "128kbps"}),
                "目标采样率Hz": (["44100Hz", "48000Hz"], {"default": "44100Hz"}),
                "强制双声道": ("BOOLEAN", {"default": True}),

                "分段时长列表": ("LIST", {"forceInput": True}),
                "原始音频": ("AUDIO", {"forceInput": True}),
                "音频开始时长": ("FLOAT", {"default": 0, "min": 0}),
            },
        }

    def align_up(self, frames, base):
        if frames <= 0:
            return 0
        return int(((frames + base - 1) // base) * base)

    def volume_to_target_db(self, vol):
        # 100→0dB，50→-6dB，1→-40dB
        return -40 + (vol / 100) * 40

    def extract_audio_segment(self, 帧率, 当前索引, 帧数对齐基数, 过渡帧数, 统一音量强度, 目标比特率kbps, 目标采样率Hz, 强制双声道, 分段时长列表, 原始音频,音频开始时长):
        print(f"✅ [凤希AI] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 开始处理音频")

        # ComfyUI音频格式：[batch, channels, samples]
        sample_rate = 原始音频["sample_rate"]
        waveform = 原始音频["waveform"]

        # 1. 开头加静音
        if 音频开始时长 > 0:
            silence_samples = int(音频开始时长 * sample_rate)
            # 适配3维张量创建静音
            silence = torch.zeros(
                (waveform.shape[0], waveform.shape[1], silence_samples),
                dtype=waveform.dtype,
                device=waveform.device
            )
            new_waveform = torch.cat([silence, waveform], dim=-1)
        else:
            new_waveform = waveform

        # 2. 统一采样率
        target_sr = int(目标采样率Hz.replace("Hz",""))
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr).to(waveform.device)
            new_waveform = resampler(new_waveform)
            sample_rate = target_sr

        # 3. 强制双声道 ✅ 修复核心代码
        if 强制双声道:
            # 获取声道数（适配3维张量）
            channels = new_waveform.size(1)
            if channels == 1:
                # 维度匹配：[batch, 1, samples] → [batch, 2, samples]
                new_waveform = new_waveform.repeat(1, 2, 1)

        # 4. 统一响度
        target_db = self.volume_to_target_db(统一音量强度)
        rms = torch.sqrt(torch.mean(new_waveform ** 2))
        if rms < 1e-8:
            gain = 1.0
        else:
            current_db = 20 * math.log10(rms.item())
            gain = 10 ** ((target_db - current_db) / 20)
        new_waveform = new_waveform * gain

        # 正确：使用最终处理完的 new_waveform 计算真实总时长
        real_total_samples = new_waveform.size(-1)
        total_seconds = real_total_samples / sample_rate

        分段时长 = [float(s) for s in 分段时长列表]
        
        if 当前索引 >= 0 and 当前索引 < len(分段时长):
            时长 = 分段时长[当前索引]
            if 时长 > total_seconds:
               total_seconds = 时长

        total_frames = total_seconds * 帧率
        aligned_frames = self.align_up(total_frames, 帧数对齐基数)
        生成帧数 = int(aligned_frames + 过渡帧数)

        # 6. 绑定统一码率标记，后续导出/合成自动用这个码率
        bitrate = int(目标比特率kbps.replace("kbps","")) * 1000
        处理后音频 = {
            "waveform": new_waveform,
            "sample_rate": sample_rate,
            "bitrate": bitrate
        }

        print(f"✅ [凤希AI] 音频处理完成 | 码率:{目标比特率kbps} | 采样率:{目标采样率Hz} | 生成帧数:{生成帧数}")
        return (处理后音频, 生成帧数)