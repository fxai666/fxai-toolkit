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

    # 返回：区间数据、循环次数、开始序号
    RETURN_TYPES = ("LIST", "INT", "INT")
    RETURN_NAMES = ("提示词数据", "循环次数", "开始序号")
    FUNCTION = "execute"
    CATEGORY = "凤希AI/图片"

    def execute(self, lines_data, 开始序号=0, 结束序号=0, 刷新标记=0):
        # 解析数据
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
        # 逻辑：结束序号为0 → 取全部数据
        if 结束序号 == 0:
            selected_lines = lines
            loop_count = total_lines
        else:
            # 截取 开始序号 ~ 结束序号（都包含自身）
            # 防止序号越界
            start = max(0, 开始序号)
            end = min(结束序号, total_lines - 1)
            selected_lines = lines[start:end+1]  # 切片不包含end，所以+1
            # 循环次数 = 结束 - 开始 + 1（包含两端）
            loop_count = end - start + 1

        return (selected_lines, loop_count, 开始序号)