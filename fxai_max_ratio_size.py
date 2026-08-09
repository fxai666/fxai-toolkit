# 纯数值计算 · 最大边长约束比例尺寸计算器
def calculate_size_by_ratio(target_max, ratio_w, ratio_h, base=32):
    scale = target_max / max(ratio_w, ratio_h)
    w = int(round(ratio_w * scale))
    h = int(round(ratio_h * scale))

    fw = (w // base) * base
    fh = (h // base) * base

    return fw, fh

class FxAiMaxRatioSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "最大边长": ("INT", {"default": 1300, "min": 64, "max": 4096, "step": 1}),
                "宽比例": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                "高比例": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
                "对齐基数": ("INT", {"default": 32, "min": 2, "max": 128, "step": 1}),
                "尺寸反转": ("BOOLEAN", {"default": False, "tooltip": "开启后宽高互换输出"}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("宽度", "高度", "对齐基数")
    FUNCTION = "process"
    CATEGORY = "凤希AI/工具"

    def process(self, 最大边长, 宽比例, 高比例, 对齐基数, 尺寸反转=False):
        try:
            out_w, out_h = calculate_size_by_ratio(最大边长, 宽比例, 高比例, 对齐基数)
            if 尺寸反转:
                out_w, out_h = out_h, out_w
            return (out_w, out_h, 对齐基数)
        except Exception as e:
            raise RuntimeError(f"计算失败：{str(e)}")