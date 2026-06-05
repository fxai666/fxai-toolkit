class FxAiInputInt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "输入数字": ("INT", {"default": "1"}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("输出",)
    FUNCTION = "convert"
    CATEGORY = "凤希AI/工具"

    def convert(self, 输入数字):
        return (输入数字,)