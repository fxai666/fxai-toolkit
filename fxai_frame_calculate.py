import math

class FxAiFrameCalculator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "秒数": ("FLOAT", {"default": 15.0}),
                "帧率": ("INT", {"default": 24}),
                "对齐基数": ("INT", {"default": 8}),
                "过渡帧": ("INT", {"default": 1}),
                "对齐方式": (["向上取整","向下取整"], {"default": "向上取整"}),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT","INT")
    RETURN_NAMES = ("最终总帧数", "帧率(小数)", "帧率(整数)")
    FUNCTION = "calculate"
    CATEGORY = "凤希AI/工具"

    def calculate(self, 秒数, 帧率, 对齐基数, 过渡帧, 对齐方式):
        raw_frame = 秒数 * 帧率
        
        if 对齐方式 == "向上取整":
            aligned_frame = math.ceil(raw_frame / 对齐基数) * 对齐基数
        else:
            aligned_frame = math.floor(raw_frame / 对齐基数) * 对齐基数
        
        aligned_frame += 过渡帧
        
        return (int(aligned_frame), float(帧率), 帧率)