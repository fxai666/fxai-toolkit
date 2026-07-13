class FxAiStrToNumber:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "输入数字": ("STRING", {"default": "1"}),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("输出整数", "输出小数")
    FUNCTION = "convert"
    CATEGORY = "凤希AI/工具"

    def convert(self, 输入数字):
        try:
            return (int(输入数字), float(输入数字))
        except ValueError:
            return (0, 0.0)