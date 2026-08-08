# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# MiniMax H3 专用音频分段加载：把长音频按分段时长列表切出当前段，
# 帧数按 H3 的 17k+5 网格对齐（帧率固定 24fps），供长视频逐段循环生成使用。

import datetime

FPS = 24


def align_down_h3(frames):
    # 最大的 17k+5 <= frames（H3 合法帧数，向下取整，剩余的留到下一段）
    if frames < 5:
        return 5
    return 17 * ((frames - 5) // 17) + 5


def align_up_h3(frames):
    # 最小的 17k+5 >= frames
    if frames <= 5:
        return 5
    k = (frames - 5 + 16) // 17
    return 17 * k + 5


class FxAiMiniMaxAudioSegmentLoad:
    CATEGORY = "凤希AI/MiniMax"
    FUNCTION = "audio_segment"

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("剪切音频", "生成帧数")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "当前索引": ("INT", {"default": 0, "min": 0}),
                "分段时长列表": ("LIST", {"forceInput": True}),
                "原始音频": ("AUDIO", {"forceInput": True}),
                "过渡帧数": ("INT", {"default": "0","step":17,"max":17}),
            },
        }

    def audio_segment(self, 当前索引, 分段时长列表, 原始音频,过渡帧数):
        print(f"✅ [凤希AI] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 开始渲染第 {当前索引+1} 个场景")
        分段时长 = [float(s) for s in 分段时长列表]
        分段数量 = len(分段时长)
        if 分段数量 == 0 or 当前索引 >= 分段数量:
            return (原始音频, 5)

        结束索引 = len(分段时长) - 1
        if 当前索引 < 0:
            raise ValueError(f"当前索引({当前索引}) 超出分段范围 0 ~ {结束索引}")

        分段对齐帧数 = [align_down_h3(round(时长 * FPS)) for 时长 in 分段时长]

        总理论帧数 = round(sum(分段时长) * FPS)
        分段对齐帧数[结束索引] = 总理论帧数 - sum(分段对齐帧数[:结束索引])

        生成帧数 = 分段对齐帧数[当前索引]

        前面帧数 = sum(分段对齐帧数[:当前索引])
        sample_rate = 原始音频["sample_rate"]
        waveform = 原始音频["waveform"]
        total_samples = waveform.size(-1)
        start_sample = max(0, min(int(前面帧数 / FPS * sample_rate), total_samples))
        end_sample = max(start_sample, min(int((前面帧数 + 生成帧数 + 过渡帧数 + 1) / FPS * sample_rate), total_samples))
        剪切音频 = {"waveform": waveform[..., start_sample:end_sample], "sample_rate": sample_rate}

        return (剪切音频, 生成帧数 + 过渡帧数)