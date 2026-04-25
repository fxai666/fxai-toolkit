import json

class FxAiMultiLineText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":{
                "lines_data": ("STRING", {"default": "[]", "multiline": True}),
            },
            "optional":{
                "循环初始值": ("INT", {"default": 0, "min": 0}),
                "刷新标记": ("INT", {"forceInput": True}),
            }
        }

    # 仅返回所有文本拼接结果和总行数
    RETURN_TYPES = ("LIST", "INT", "INT")
    RETURN_NAMES = ("提示词数据", "总行数","循环初始值")
    FUNCTION = "execute"
    CATEGORY = "凤希AI"

    def execute(self, lines_data,循环初始值=0,刷新标记=0):
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

        return (lines, len(lines),循环初始值)