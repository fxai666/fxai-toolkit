class FxAiSceneLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "场景数据": ("LIST", {"forceInput": True}),
                "行索引": ("INT", {"forceInput": True}),
                "循环复用": ("INT", {"default": 0, "min": 0}),
            },
            "optional": {
                "刷新标记": ("INT", {"forceInput": True}),
                "通用提示词": ("STRING", {"default": "", "forceInput": True}),
                "尾部通用提示词": ("STRING", {"default": "", "forceInput": True}),
                "默认时长": ("INT", {"default":15, "min": 1}),
            }
        }

    # 输出：新增「尾帧位置」端口
    RETURN_TYPES = ("FLOAT", "STRING", "INT", "INT", "INT", "BOOLEAN", "INT")
    RETURN_NAMES = (
        "时长(秒)",
        "提示词",
        "音频索引",
        "图片索引",
        "尾帧位置",
        "启用转场",
        "场景行索引"
    )

    FUNCTION = "get_scene_data"
    CATEGORY = "凤希AI/场景管理"

    def get_scene_data(self, 场景数据, 行索引,循环复用, 刷新标记=0,通用提示词="",尾部通用提示词="",默认时长=15):
        # 安全校验
        if not isinstance(场景数据, list) or len(场景数据) == 0:
            return (默认时长, f"{通用提示词}{尾部通用提示词}", 行索引, 行索引, -1, True, 行索引)
    
        if 循环复用 > 1:
           行索引 = 行索引 % 循环复用
        elif 循环复用 == 1:
             行索引 = 0

        # 索引越界自动修正
        total_lines = len(场景数据)
        if 行索引 >= total_lines:
            return (默认时长, f"{通用提示词}{尾部通用提示词}", 行索引, 行索引, -1, True, 行索引)

        # 取出指定行数据
        line = 场景数据[行索引]

        # 解析字段
        时长 = float(line.get("时长", 5.0))
        提示词 = f"{通用提示词}{line.get('提示词文本', '')}{尾部通用提示词}"
        音频索引 = int(line.get("音频索引", 0))
        图片索引 = int(line.get("图片索引", -1))
        if 图片索引 < 0:
           图片索引 = 行索引
        
        启用转场 = int(line.get("转场", 1)) == 1
        
        尾帧位置 = line.get("尾帧位置", -1)

        return (时长, 提示词, 音频索引, 图片索引, 尾帧位置, 启用转场, 行索引)