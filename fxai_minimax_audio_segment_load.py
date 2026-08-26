# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# MiniMax H3 专用音频分段加载：把长音频按分段时长列表切出当前段。
# 分段时长列表由 fxai_audio_segments_v2（目标模型=MiniMaxH3）按 17k+5 对齐后输出，
# 每段已是真正的生成时长，这里直接换算帧数切片，不再重复对齐。

import datetime

FPS = 24


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

        # 分段时长已是对齐后的每段真正时长，乘以 24 即每段帧数，逐段累加切片
        分段帧数 = [int(round(时长 * FPS)) for 时长 in 分段时长]
        if any(f > 24000 for f in 分段帧数):
            raise ValueError(
                f"分段时长列表疑似传入了帧数而非秒数（某段换算帧数 {max(分段帧数)} > 24000 = 1000 秒）。"
                f"请接入 fxai_audio_segments_v2 输出的秒数分段列表，当前值：{分段时长}")

        生成帧数 = 分段帧数[当前索引]

        前面帧数 = sum(分段帧数[:当前索引])
        sample_rate = 原始音频["sample_rate"]
        waveform = 原始音频["waveform"]
        total_samples = waveform.size(-1)
        start_sample = max(0, min(int(前面帧数 / FPS * sample_rate), total_samples))
        end_sample = max(start_sample, min(int((前面帧数 + 生成帧数 + 过渡帧数 + 1) / FPS * sample_rate), total_samples))
        剪切音频 = {"waveform": waveform[..., start_sample:end_sample], "sample_rate": sample_rate}

        return (剪切音频, 生成帧数 + 过渡帧数)