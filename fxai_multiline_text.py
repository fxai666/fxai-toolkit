import json

class FxAiMultiLineText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lines_data": ("STRING", {"default": "[]", "multiline": True}),
            },
            "optional": {
                "开始序号": ("INT", {"default": 0, "min": 0}),
                "结束序号": ("INT", {"default": 0, "min": 0}),
                "刷新标记": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("LIST", "INT", "INT")
    RETURN_NAMES = ("提示词数据", "循环次数", "开始索引")
    FUNCTION = "execute"
    CATEGORY = "凤希AI/图片"

    def execute(self, lines_data, 开始序号=0, 结束序号=0, 刷新标记=0):
        try:
            if isinstance(lines_data, str):
                lines = json.loads(lines_data.strip())
            elif isinstance(lines_data, list):
                lines = lines_data
            else:
                lines = []
        except Exception as e:
            lines = []
            print(f"解析lines_data失败: {e}")

        total_lines = len(lines)
        start_idx = 开始序号

        loop_count = total_lines - start_idx

        if 结束序号 > 0 and 结束序号 > 开始序号:
            loop_count = 结束序号 - 开始序号
        elif 结束序号 == 开始序号 != 0:
            loop_count = 1

        loop_count = max(loop_count, 0)
        return (lines, loop_count, max(start_idx - 1,0))