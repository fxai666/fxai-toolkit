import json

class FxaiStoryBoard:
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
    RETURN_NAMES = ("分镜数据", "循环次数", "开始索引")
    FUNCTION = "execute"
    CATEGORY = "凤希AI/图片"

    def execute(self, lines_data, 开始序号=0, 结束序号=0, 刷新标记=0):
        try:
            lines = json.loads(lines_data.strip())
        except:
            lines = []

        total_lines = len(lines)
        开始索引 = max(开始序号 - 1, 0)

        loop_count = total_lines - 开始索引

        if 结束序号 > 0 and 结束序号 > 开始序号:
            loop_count = 结束序号 - 开始序号
        elif 结束序号 == 开始序号 != 0:
            loop_count = 1

        loop_count = max(loop_count, 1)
        
        return (lines, loop_count, 开始索引)