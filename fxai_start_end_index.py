class FxAiStartEndIndex:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "开始索引": ("INT", {"default": 0, "min": 0, "max": 999}),
                "结束索引": ("INT", {"default": -1, "min": -1, "max": 999}),
                "总循环数": ("INT", {"default": 0, "min":1, "max": 999}),
            },
        }

    RETURN_TYPES = ("INT","INT","INT")
    RETURN_NAMES = ("开始索引","结束索引","总循环数")
    FUNCTION = "convert"
    CATEGORY = "凤希AI/工具"

    def convert(self, 开始索引,结束索引,总循环数):
        if 结束索引 > -1 and 结束索引 >= 开始索引:
           总循环数 = min(结束索引 - 开始索引 + 1,总循环数)
        return (开始索引,结束索引,总循环数)