class FxAiStrToNumber:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "输入数字字符串": ("STRING", {"default": "1"}),
                "保留小数位数": ("INT", {"default": 2, "min": 0, "max": 10}),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("输出整数", "输出小数")
    FUNCTION = "convert"
    CATEGORY = "凤希AI/工具"

    def convert(self, 输入数字字符串, 保留小数位数):
        num = float(输入数字字符串)
        
        输出整数 = int(num)
        
        if 保留小数位数 <= 0:
            输出小数 = float(int(num))
        else:
            输出小数 = round(num, 保留小数位数)
        
        return (输出整数, 输出小数)