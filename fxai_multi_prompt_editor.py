import json
from typing import List, Dict, Any

class FxAiMultiPromptEditor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts_data": ("STRING", {"multiline": True, "default": "[]"}),
            },
            "optional": {
                "刷新标记": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("多提示词数据",)
    FUNCTION = "execute"
    CATEGORY = "凤希AI/场景管理"

    def execute(self, prompts_data="[]", 刷新标记=0) -> tuple[List[Dict[str, Any]]]:
        try:
            if isinstance(prompts_data, str):
                prompts = json.loads(prompts_data.strip())
            elif isinstance(prompts_data, (list, dict)):
                prompts = prompts_data
            else:
                prompts = []
        except Exception as e:
            print(f"[凤希AI] 解析提示词数据失败: {e}")
            prompts = []

        prompt_data = []
        for idx, line in enumerate(prompts):
            line_data = {
                "序号": idx + 1,
                "索引编号": 0,
                "开始时间": 0.0,
                "结束时间": 15.0,
                "提示词文本": "",
            }

            try:
                line_data["索引编号"] = int(line.get("索引编号", 0))
                line_data["开始时间"] = float(line.get("开始时间", 0.0))
                line_data["结束时间"] = float(line.get("结束时间", 15.0))
                line_data["提示词文本"] = line.get("提示词文本", "")
            except:
                pass

            prompt_data.append(line_data)

        return (prompt_data,)