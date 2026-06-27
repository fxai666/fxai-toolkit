import datetime
class FxAIAudioFrameCalculate:
    CATEGORY = "凤希AI/音频"
    FUNCTION = "calculate"

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("生成帧数",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "音频": ("AUDIO", {"forceInput": True}),
                "帧率": ("INT", {"default": 24, "min": 1}),
                "帧数对齐基数": ("INT", {"default": 8, "min": 1}),
                "过渡帧数": ("INT", {"default": 1, "min": 0}),
            },
        }

    # 向上对齐（返回纯整数）
    def align_up(self, frames, base):
        if frames <= 0:
            return 0
        return int(((frames + base - 1) // base) * base)

    def calculate(self, 音频, 帧率, 帧数对齐基数, 过渡帧数):
        print(f"✅ [凤希AI] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 读取完整音频")

        sample_rate = 音频["sample_rate"]
        waveform = 音频["waveform"]
        total_samples = waveform.size(-1)

        total_seconds = total_samples / sample_rate
        原始总帧数 = int(total_seconds * 帧率)
        对齐总帧数 = self.align_up(原始总帧数, 帧数对齐基数)

        生成帧数 = int(对齐总帧数 + 过渡帧数)
        return (生成帧数,)