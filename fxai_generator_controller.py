class FxAIGeneratorController:
    """
    纯数据输出控制器（无任何对齐、帧数、音频计算）
    功能：仅计算 开始索引、循环次数
    输出：分段时长列表 + 开始索引 + 循环数 + 帧率 + 宽度 + 高度
    """
    CATEGORY = "凤希AI/工具"
    FUNCTION = "process"

    # 返回值：增加了处理后的宽度、高度
    RETURN_TYPES = ("LIST", "INT", "INT", "INT", "INT", "INT", "INT","INT", "INT","INT")
    RETURN_NAMES = (
        "分段时长列表",
        "开始索引",
        "循环数",
        "每秒帧数",
        "实际宽度",
        "实际高度",
        "分段数量",
        "过渡帧数",
        "实际半宽",
        "实际半高",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "启用场景分段": ("BOOLEAN", {"default": False}),
                "开始索引": ("INT", {"default": 0, "min": 0}),
                "结束索引": ("INT", {"default": 0, "min": 0}),
                "帧率": ("INT", {"default": 24, "min": 1}),
                "宽度": ("INT", {"default": 960, "min": 544, "step": 2}),
                "高度": ("INT", {"default": 544, "min": 544, "step": 2}),
                "长宽对齐基数": ("INT", {"default": 32, "min": 1}),
            },
            "optional": {
                "场景分段时长": ("LIST", {"forceInput": True}),
                "音频分段时长": ("LIST", {"forceInput": True}),
                "过渡帧数": ("INT", {"default": 9, "min": 1,"step":8}),
                "采样次数": ("INT", {"default": 1, "min": 1,"step":1}),
            }
        }

    def process(self, 启用场景分段, 开始索引, 结束索引, 帧率, 宽度, 高度, 长宽对齐基数, 场景分段时长=None, 音频分段时长=None,过渡帧数=9,采样次数=1):
        # 1. 纯读取分段时长，不做任何计算
        if 启用场景分段 and 场景分段时长 is not None:
            分段时长 = 场景分段时长
        else:
            分段时长 = 音频分段时长 if 音频分段时长 is not None else []			

        if 长宽对齐基数 > 1:
            长宽对齐基数 = 长宽对齐基数 * 采样次数
            最终宽度 = 宽度 // 长宽对齐基数 * 长宽对齐基数
            最终高度 = 高度 // 长宽对齐基数 * 长宽对齐基数
        else:
            最终宽度 = 宽度
            最终高度 = 高度

        分段数量 = len(分段时长)
        if 分段数量 == 0:
            return ([], 0, 1, 帧率, 最终宽度, 最终高度, 0, 过渡帧数, 最终宽度//2, 最终高度//2)

        开始索引 = max(0, min(开始索引, 分段数量 - 1))

        if 结束索引 > 0:
           结束索引 = 结束索引
        else:
           结束索引 = 分段数量

        结束索引 = max(开始索引, 结束索引)

        循环数 = max(结束索引 - 开始索引,1)

        return (分段时长, 开始索引, 循环数, 帧率, 最终宽度, 最终高度,分段数量,过渡帧数,最终宽度//2,最终高度//2)