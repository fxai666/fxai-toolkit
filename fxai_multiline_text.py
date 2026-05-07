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
        # 安全处理：开始序号不能超过总行数
        start_idx = min(开始序号, total_lines)
        
        # ===================== 核心逻辑修改 =====================
        # 规则1：结束序号 <= 开始序号 → 循环次数 = 总行数 - 开始序号
        if 结束序号 < 开始序号:
            loop_count = total_lines - start_idx
        # 规则2：结束序号 > 开始序号 → 循环次数 = 结束序号 - 开始序号
        else:
            loop_count = 结束序号 - 开始序号
        
        # 循环次数不能为负数
        loop_count = max(1, loop_count)
        # ======================================================

        # 始终返回 完整原始数据列表，不截取
        return (lines, loop_count, 开始序号)