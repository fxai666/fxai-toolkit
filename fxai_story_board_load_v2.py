import re

class FxaiStoryBoardLoadV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "分镜数据": ("LIST", {"forceInput": True}),
                "行索引": ("INT", {"forceInput": True}),
            },
            "optional": {
                "通用提示词": ("STRING", {"default": "", "forceInput": True}),
                "尾部通用提示词": ("STRING", {"default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("提示词", "行索引", "角色资源")
    FUNCTION = "get_scene_data"
    CATEGORY = "凤希AI/分镜"

    def get_scene_data(self, 分镜数据, 行索引, 刷新标记=0, 通用提示词="", 尾部通用提示词=""):

        if not isinstance(分镜数据, list) or len(分镜数据) == 0:
            return (f"{通用提示词}{尾部通用提示词}", 行索引, "")

        total_lines = len(分镜数据)
        if 行索引 < 0 or 行索引 >= total_lines:
            return (f"{通用提示词}{尾部通用提示词}", 行索引, "")

        line_item = 分镜数据[行索引]
        scene_prop = line_item.get("角色资源", "")

        line_text = line_item.get("提示词", "")
        final_prompt = f"{通用提示词}{line_text}{尾部通用提示词}"

        return (final_prompt, 行索引, scene_prop)