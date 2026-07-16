import json

class FxAiMultiLinePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lines_data": ("STRING", {"default": "[]", "multiline": True}),
                "当前索引": ("INT", {"forceInput": True, "default": 0}),
            },
            "optional": {
                "通用提示词": ("STRING", {"forceInput": True}),
                "尾部通用提示词": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("提示词",)
    FUNCTION = "execute"
    CATEGORY = "凤希AI/提示词"

    def execute(self, lines_data, 当前索引,通用提示词="",尾部通用提示词=""):
        # 解析json数组字符串
        try:
            if isinstance(lines_data, str):
                data_str = lines_data.strip()
                lines = json.loads(data_str)
            elif isinstance(lines_data, list):
                lines = lines_data
            else:
                lines = []
        except Exception:
            lines = []

        total_lines = len(lines)
        if total_lines == 0:
            return (f"{通用提示词}{尾部通用提示词}",)
        
        safe_index = 当前索引 % total_lines
        result_text = lines[safe_index]
        return (f"{通用提示词}{str(result_text)}{尾部通用提示词}",)
		
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")