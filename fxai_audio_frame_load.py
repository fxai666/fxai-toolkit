import datetime
class FxAIAudioSegmentLoad:
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

                "分段时长列表": ("LIST", {"forceInput": True}),
                "原始音频": ("AUDIO", {"forceInput": True}),
            },
        }

    # 向下对齐（返回纯整数）
    def align_down(self, frames, base):
        return int(frames // base * base)

    # 向上对齐（返回纯整数）
    def align_up(self, frames, base):
        if frames <= 0:
            return 0
        return int(((frames + base - 1) // base) * base)

    def extract_audio_segment(self, 帧率, 当前索引, 帧数对齐基数, 过渡帧数, 分段时长列表, 原始音频):
        print(f"✅ [凤希AI] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 开始渲染第 {当前索引+1} 个场景")
        # 1. 基础数据转换
        分段时长 = [float(s) for s in 分段时长列表]
        分段数量 = len(分段时长)

        if 分段数量 == 0:
            return (原始音频, int(过渡帧数))

        结束索引 = 分段数量 - 1

        if 当前索引 < 0 or 当前索引 > 结束索引:
            raise ValueError(f"❌ 当前索引({当前索引}) 超出分段有效范围！允许范围：0 ~ {结束索引}")

        分段原始帧数 = [int(时长 * 帧率) for 时长 in 分段时长]
        分段对齐帧数 = [self.align_down(f, 帧数对齐基数) for f in 分段原始帧数]

        总理论帧数 = int(sum(分段时长) * 帧率)
        总对齐帧数 = sum(分段对齐帧数)
        缺失帧数 = 总理论帧数 - 总对齐帧数

        分段对齐帧数[结束索引] = self.align_up(分段对齐帧数[结束索引] + 缺失帧数, 帧数对齐基数)
        分段时长[结束索引] = 分段对齐帧数[结束索引] / 帧率
        前面总对齐帧数 = sum(分段对齐帧数[:当前索引])
        实际开始秒 = 前面总对齐帧数 / 帧率

        sample_rate = 原始音频["sample_rate"]
        waveform = 原始音频["waveform"]

        total_samples = waveform.size(-1)  # 永远取最后一维 = 总采样数（通用所有维度）

        # ========== 100% 保留你原来的计算逻辑 ==========
        start_sample = int(实际开始秒 * sample_rate)
        end_sample = start_sample + int((分段时长[当前索引] + 过渡帧数/帧率 + 0.5) * sample_rate)

        # ========== 安全边界（修复空音频问题） ==========
        start_sample = max(0, min(start_sample, total_samples))
        end_sample = max(start_sample, min(end_sample, total_samples))

        截取后音频_data = waveform[..., start_sample:end_sample]  # ... 代表所有前面维度，通用1/2/3维
        截取后音频 = {
            "waveform": 截取后音频_data,
            "sample_rate": sample_rate
        }

        生成帧数 = int(分段对齐帧数[当前索引] + 过渡帧数)

        return (截取后音频, 生成帧数)