class FxAiStartEndIndex:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "开始索引": ("INT", {"default": 0, "min": 0, "max": 999}),
                "结束索引": ("INT", {"default": -1, "min": -1, "max": 999}),
                "总循环数": ("INT", {"default": 10, "min": 1, "max": 999}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("开始索引", "结束索引", "总循环数")
    FUNCTION = "convert"
    CATEGORY = "凤希AI/工具"

    def convert(self, 开始索引, 结束索引, 总循环数):
        开始索引 = max(0, min(开始索引, 总循环数 - 1))
        
        if 结束索引 < 0:
            结束索引 = 总循环数 - 1
        
        结束索引 = max(开始索引, min(结束索引, 总循环数 - 1))
        
        real_loop_count = 结束索引 - 开始索引 + 1
        
        return (开始索引, 结束索引, real_loop_count)