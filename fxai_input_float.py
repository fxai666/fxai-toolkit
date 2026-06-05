class FxAiInputFloat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "输入数字": ("FLOAT", {"default": "0"}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("输出",)
    FUNCTION = "convert"
    CATEGORY = "凤希AI/工具"

    def convert(self, 输入数字):
        return (输入数字,)