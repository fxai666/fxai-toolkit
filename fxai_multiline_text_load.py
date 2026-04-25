class FxAiMultiLineTextLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词数据": ("LIST", {"forceInput": True}),  # 接收上游的列表
                "行索引": ("INT", {"forceInput": True}),
                "循环复用": ("INT", {"default": 0, "min": 0}),
            },
            "optional": {
                "刷新标记": ("INT", {"forceInput": True}),
                "通用提示词": ("STRING", {"default": "", "forceInput": True}),
                "尾部通用提示词": ("STRING", {"default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING","INT")
    RETURN_NAMES = ("提示词", "行索引")
    FUNCTION = "get_scene_data"
    CATEGORY = "凤希AI"

    # 修正：补充循环复用参数，修复变量作用域问题
    def get_scene_data(self, 提示词数据, 行索引, 循环复用, 刷新标记=0,通用提示词="",尾部通用提示词=""):
        # 安全校验：确保是有效列表
        if not isinstance(提示词数据, list) or len(提示词数据) == 0:
            return (f"{通用提示词}{尾部通用提示词}", 行索引)
    
        # 循环复用处理：索引取模
        if 循环复用 > 1:
           行索引 = 行索引 % 循环复用
        elif 循环复用 == 1:
             行索引 = 0

        total_lines = len(提示词数据)
        if 行索引 >= total_lines or 行索引 < 0:
            return (f"{通用提示词}{尾部通用提示词}", 行索引)

        line_text = 提示词数据[行索引]

        # 拼接提示词
        提示词 = f"{通用提示词}{line_text}{尾部通用提示词}"

        return (提示词, 行索引)