class FxAiFrameCalculator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "秒数": ("FLOAT", {"default": 15.0}),
                "帧率": ("INT", {"default": 24}),
                "对齐基数": ("INT", {"default": 8}),
                "偏移量": ("INT", {"default": 1}),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT","INT")
    RETURN_NAMES = ("最终总帧数", "帧率(小数)", "帧率(整数)")
    FUNCTION = "calculate"
    CATEGORY = "凤希AI/工具"

    def calculate(self, 秒数, 帧率, 对齐基数, 偏移量):
        raw_frame = 秒数 * 帧率
        aligned_frame = (raw_frame // 对齐基数) * 对齐基数 + 偏移量
        
        # 输出：最终帧数、帧率
        return (int(aligned_frame),float(帧率), 帧率)