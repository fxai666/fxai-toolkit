class FxAiSceneLoadV2:
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
            }
        }

    RETURN_TYPES = ("INT", "STRING", "INT", "FLOAT", "FLOAT", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("场景行索引","提示词","音频文件索引","音频开始(秒)","音频时长(秒)","图片索引","尾帧位置","启用转场")

    FUNCTION = "get_scene_data"
    CATEGORY = "凤希AI/场景管理"

    def get_scene_data(self, 场景数据, 行索引, 循环复用, 刷新标记=0, 通用提示词="", 尾部通用提示词=""):
        try:
            if 循环复用 > 1:
                行索引 = 行索引 % 循环复用
            elif 循环复用 == 1:
                行索引 = 0

            total_lines = len(场景数据) if isinstance(场景数据, list) else 0

            # ====================== 异常判断：越界 / 无数据
            if 行索引 < 0 or total_lines == 0 or 行索引 >= total_lines:
                # 主动抛出异常，附带信息
                raise IndexError(
                    f"• 当前行索引：{行索引}\n"
                    f"• 场景总行数：{total_lines}\n"
                )

            # ====================== 正常读取数据
            item = 场景数据[行索引]
            
            提示词 = 通用提示词 + item["提示词文本"] + 尾部通用提示词
            音频开始 = float(item["音频开始"])
            音频时长 = float(item["音频时长"])
            音频索引 = int(item["音频索引"])
            图片索引 = int(item["图片索引"])
            尾帧位置 = int(item["尾帧位置"])
            启用转场 = int(item["转场"]) == 1

            if 图片索引 < 0:
                图片索引 = 行索引

            if 音频索引 < 0:
                音频索引 = 行索引

            # 正常返回
            return (行索引, 提示词, 音频索引, 音频开始, 音频时长, 图片索引, 尾帧位置, 启用转场)

        # ====================== 异常捕获 + 打印 + 返回默认值
        except Exception as e:
            print(f"✅ [凤希AI场景] 已加载默认值。信息： \n{e}")
            
            提示词 = f"{通用提示词}{尾部通用提示词}"
            return (行索引, 提示词, 行索引, 0.0, 15, 行索引, -1, True)