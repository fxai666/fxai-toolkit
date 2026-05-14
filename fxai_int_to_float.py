class FxAiIntToFloat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "输入整数": ("INT", {"default": 0, "min": -9999999, "max": 9999999}),
            }
        }

    RETURN_TYPES = ("FLOAT","INT")
    RETURN_NAMES = ("小数","整数",)
    FUNCTION = "convert"
    CATEGORY = "凤希AI/工具"

    def convert(self, 输入整数):
        # 直接把整数转成小数
        输出小数 = float(输入整数)
        return (输出小数,输入整数)