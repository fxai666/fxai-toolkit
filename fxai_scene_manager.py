import json
from typing import List, Dict, Any

class FxAiSceneManager:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lines_data": ("STRING", {"default": "[]", "multiline": True}),
            },
            "optional":{
                "刷新标记": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("INT", "LIST")
    RETURN_NAMES = ("总行数", "场景数据")
    FUNCTION = "execute"
    CATEGORY = "凤希AI"

    def execute(self, lines_data="[]", 刷新标记=0) -> tuple[int, List[Dict[str, Any]]]:
        try:
            if isinstance(lines_data, str):
                lines = json.loads(lines_data.strip())
            elif isinstance(lines_data, (list, dict)):
                lines = lines_data
            else:
                lines = []
        except Exception as e:
            lines = []
            print(f"解析lines_data失败: {e}")

        total_count = len(lines)
        scene_data = []

        for idx, line in enumerate(lines):
            line_data = {
                "序号": idx + 1,
                "时长": 5.0,
                "提示词文本": "",
                "音频索引": 0,
                "图片索引": -1,
                "转场": 1,
                "尾帧位置": -1
            }

            # ✅ 最干净、最安全、不会丢0 的写法
            if isinstance(line, list):
                if len(line) >= 1: line_data["时长"] = float(line[0])
                if len(line) >= 2: line_data["提示词文本"] = line[1]
                if len(line) >= 3: line_data["音频索引"]   = int(line[2])
                if len(line) >= 4: line_data["图片索引"]   = int(line[3])
                if len(line) >= 5: line_data["尾帧位置"]   = int(line[4])
                if len(line) >= 6: line_data["转场"]       = int(line[5])

            elif isinstance(line, str):
                line_data["提示词文本"] = line

            scene_data.append(line_data)

        return (total_count, scene_data)