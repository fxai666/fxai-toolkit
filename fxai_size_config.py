class FxAiSizeConfig:
    """
    宽高设置器（带反转开关）
    勾选反转后，自动交换宽高
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "宽度": ("INT", {
                    "default": 1080,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "display": "number"
                }),
                "高度": ("INT", {
                    "default": 1920,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "display": "number"
                }),
                "反转": ("BOOLEAN", {
                    "default": False,
                    "label_on": "反转 (宽↔高)",
                    "label_off": "正常"
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("最终宽度", "最终高度")
    FUNCTION = "process"
    CATEGORY = "凤希AI/工具"

    def process(self, 宽度, 高度, 反转):
        if 反转:
            return (高度,宽度)
        else:
            return (宽度, 高度)