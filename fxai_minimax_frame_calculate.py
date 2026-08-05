# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# MiniMax H3 专用帧数计算：按音频时长向下对齐到 17k+5 网格。
# 生成时长不会超过音频（剩余声音留给下一个循环段继续取），
# 帧率固定 24fps（H3 模型固定，不可改 30）。

import datetime

FPS = 24
AUDIO_LATENT_FPS = 40


def align_down_h3(frames):
    # 最大的 17k+5 <= frames（H3 合法帧数网格）
    if frames < 5:
        return 5
    return 17 * ((frames - 5) // 17) + 5


def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


class FxAiMiniMaxFrameCalculate:
    CATEGORY = "凤希AI/MiniMax"
    FUNCTION = "calculate"

    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("生成帧数", "视频时长", "剩余音频时长", "视频潜变量T", "音频潜变量T")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "音频": ("AUDIO", {"forceInput": True}),
                "过渡帧数": ("INT", {"default": 0, "min": 0,
                    "tooltip": "额外生成的重叠帧，先加总再向下对齐到 17k+5 网格"}),
            },
        }

    def calculate(self, 音频, 过渡帧数):
        print(f"✅ [凤希AI] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} MiniMax帧数计算")

        sample_rate = 音频["sample_rate"]
        waveform = 音频["waveform"]
        total_samples = waveform.size(-1)

        原始秒数 = total_samples / sample_rate
        raw_frames = int(原始秒数 * FPS) + 过渡帧数
        生成帧数 = align_down_h3(raw_frames)
        视频时长 = 生成帧数 / FPS
        剩余音频时长 = max(0.0, 原始秒数 - 视频时长)
        视频潜变量T = video_latent_t(生成帧数)
        音频潜变量T = round(视频时长 * AUDIO_LATENT_FPS)

        return (生成帧数, 视频时长, 剩余音频时长, 视频潜变量T, 音频潜变量T)
