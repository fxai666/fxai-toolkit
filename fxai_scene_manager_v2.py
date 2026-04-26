import json
from typing import List, Dict, Any

class FxAiSceneManagerV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lines_data": ("STRING", {"multiline": True, "default": "[]"}),
            },
            "optional": {
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
        except:
            lines = []

        total_count = len(lines)
        scene_data = []

        for idx, line in enumerate(lines):
            line_data = {
                "序号": idx + 1,
                "音频时长": 5.0,
                "提示词文本": "",
                "音频索引": 0,
                "音频开始": 0.0,      # ✅ 新增：音频开始
                "图片索引": -1,
                "尾帧位置": -1,
                "转场": 1
            }

            if isinstance(line, list):
                try:
                    # 顺序完全对齐前端JS：音频时长、提示词、音频索引、音频开始、图片索引、尾帧位置、转场
                    if len(line) >= 1:
                        line_data["音频时长"] = float(line[0]) if line[0] else 5.0
                    if len(line) >= 2:
                        line_data["提示词文本"] = line[1] if line[1] else ""
                    if len(line) >= 3:
                        line_data["音频索引"] = int(line[2]) if line[2] else 0
                    if len(line) >= 4:
                        line_data["音频开始"] = float(line[3]) if line[3] else 0.0  # ✅ 解析音频开始
                    if len(line) >= 5:
                        line_data["图片索引"] = int(line[4]) if line[4] else -1
                    if len(line) >= 6:
                        line_data["尾帧位置"] = int(line[5]) if line[5] else -1
                    if len(line) >= 7:
                        line_data["转场"] = int(line[6]) if line[6] else 1
                except:
                    pass
            elif isinstance(line, str):
                line_data["提示词文本"] = line

            scene_data.append(line_data)

        return (total_count, scene_data)