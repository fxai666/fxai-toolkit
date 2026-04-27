class FxAIGeneratorController:
    """
    纯数据输出控制器（无任何对齐、帧数、音频计算）
    功能：仅计算 开始索引、循环次数
    输出：分段时长列表 + 开始索引 + 循环数 + 帧率 + 宽度 + 高度
    """
    CATEGORY = "凤希AI/其他"
    FUNCTION = "process"

    # 返回值：增加了处理后的宽度、高度
    RETURN_TYPES = ("LIST", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = (
        "分段时长列表",
        "开始索引",
        "循环数",
        "每秒帧数",
        "实际宽度",
        "实际高度",
        "分段数量"
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
            }
        }

    def process(self, 启用场景分段, 开始索引, 结束索引, 帧率, 宽度, 高度, 长宽对齐基数, 场景分段时长=None, 音频分段时长=None):
        # 1. 纯读取分段时长，不做任何计算
        if 启用场景分段 and 场景分段时长 is not None:
            分段时长 = 场景分段时长
        else:
            分段时长 = 音频分段时长 if 音频分段时长 is not None else []

        # 2. 索引安全处理
        分段数量 = len(分段时长)
        if 分段数量 == 0:
            return ([], 0, 0, 帧率, 宽度, 高度)

        开始索引 = max(0, min(开始索引, 分段数量 - 1))
        if 结束索引 < 1:
           结束索引 = 分段数量 - 1
        结束索引 = max(开始索引, 结束索引)

        # 3. 只计算循环数
        循环数 = 结束索引 - 开始索引 + 1

        # ===================== 核心新增功能 =====================
        # 宽/高向下取最接近 能被长宽对齐基数整除 的值
        if 长宽对齐基数 >= 1:
            最终宽度 = 宽度 - (宽度 % 长宽对齐基数)
            最终高度 = 高度 - (高度 % 长宽对齐基数)
        else:
            最终宽度 = 宽度
            最终高度 = 高度

        # 4. 全部原样/处理后输出
        return (
            分段时长,
            开始索引,
            循环数,
            帧率,
            最终宽度,
            最终高度,
			分段数量
        )