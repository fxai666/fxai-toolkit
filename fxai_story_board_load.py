import re

class FxaiStoryBoardLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "分镜数据": ("LIST", {"forceInput": True}),
                "行索引": ("INT", {"forceInput": True}),
                "循环复用": ("INT", {"default": 0, "min": 0}),
                "默认场景序号": ("STRING", {"default": "-1"}),
            },
            "optional": {
                "通用提示词": ("STRING", {"default": "", "forceInput": True}),
                "尾部通用提示词": ("STRING", {"default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("提示词", "行索引", "场景道具")
    FUNCTION = "get_scene_data"
    CATEGORY = "凤希AI/图片"

    # 正确正则：连续的非数字 → 替换成 一个逗号
    def convert_to_comma(self, text):
        if not text:
            return ""
        text = re.sub(r'[^\w\./]+', ',', text)
        return text

    def get_scene_data(self, 分镜数据, 行索引, 循环复用, 默认场景序号="1,2,3", 刷新标记=0, 通用提示词="", 尾部通用提示词=""):
        default_formatted = self.convert_to_comma(默认场景序号)

        if not isinstance(分镜数据, list) or len(分镜数据) == 0:
            return (f"{通用提示词}{尾部通用提示词}", 行索引, default_formatted)

        if 循环复用 > 1:
            行索引 = 行索引 % 循环复用
        elif 循环复用 == 1:
            行索引 = 0

        total_lines = len(分镜数据)
        if 行索引 < 0 or 行索引 >= total_lines:
            return (f"{通用提示词}{尾部通用提示词}", 行索引, default_formatted)

        line_item = 分镜数据[行索引]
        scene_prop = line_item.get("场景道具", "")
        scene_formatted = self.convert_to_comma(scene_prop)

        if not scene_formatted:
            scene_formatted = default_formatted

        line_text = line_item.get("提示词", "")
        final_prompt = f"{通用提示词}{line_text}{尾部通用提示词}"

        return (final_prompt, 行索引, scene_formatted)