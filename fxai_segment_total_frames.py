import datetime
class FxAiSegmentTotalFrames:
    CATEGORY = "凤希AI/工具"
    FUNCTION = "calc_align_frames"

    # 只输出：生成帧数
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("生成帧数",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "帧率": ("INT", {"default": 24, "min": 1}),
                "当前索引": ("INT", {"default": 0, "min": 0}),
                "帧数对齐基数": ("INT", {"default": 8, "min": 1}),
                "分段时长列表": ("LIST", {"forceInput": True}),
            },
        }

    # 向上对齐（8的倍数）
    def align_up(self, frames, base):
        if frames <= 0:
            return 0
        return int(((frames + base - 1) // base) * base)

    def calc_align_frames(self, 帧率, 当前索引, 帧数对齐基数, 分段时长列表):
        print(f"✅ [凤希AI] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 计算对齐帧数")

        # 1. 基础数据
        分段时长 = [float(s) for s in 分段时长列表]
        分段数量 = len(分段时长)

        if 分段数量 == 0:
            return (0,)

        结束索引 = 分段数量 - 1

        # 索引校验
        if 当前索引 < 0 or 当前索引 > 结束索引:
            raise ValueError(f"❌ 当前索引({当前索引}) 超出分段有效范围！允许范围：0 ~ {结束索引}")

        # 2. 只取当前索引对应的时长 → 转帧数 → 向上对齐
        当前分段时长 = 分段时长[当前索引]
        当前分段帧数 = int(当前分段时长 * 帧率)
        
        生成帧数 = self.align_up(当前分段帧数, 帧数对齐基数) + 1

        return (生成帧数,)