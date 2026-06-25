class FxAiMultiLineTextGrid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词数据": ("LIST", {"forceInput": True}),
            },
            "optional": {
                "通用提示词": ("STRING", {"default": "", "forceInput": True}),
                "尾部通用提示词": ("STRING", {"default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("提示词",)
    FUNCTION = "get_scene_data"
    CATEGORY = "凤希AI/图片"

    def get_scene_data(self, 提示词数据, 通用提示词="", 尾部通用提示词=""):
        # 空列表兜底
        if not isinstance(提示词数据, list) or len(提示词数据) == 0:
            return (f"{通用提示词}{尾部通用提示词}",)

        total_lines = len(提示词数据)
        grid_prefix = f"{total_lines}宫格画面，多格分镜布局。\n"
        line_content_list = []
        for 行索引 in range(total_lines):
            line_content_list.append(f"场景{行索引+1}：{str(提示词数据[行索引])}")
        all_line_text = "\n".join(line_content_list)

        return (f"{grid_prefix}{通用提示词}{all_line_text}{尾部通用提示词}",)