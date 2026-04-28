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

    RETURN_TYPES = ("INT", "LIST", "LIST")
    RETURN_NAMES = ("总行数", "场景数据", "分段时长")
    FUNCTION = "execute"
    CATEGORY = "凤希AI/场景管理"

    def execute(self, lines_data="[]", 刷新标记=0) -> tuple[int, List[Dict[str, Any]], List[float]]:
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
        segment_durations = []

        for idx, line in enumerate(lines):
            line_data = {
                "序号": idx + 1,
                "音频时长": 15.0,
                "提示词文本": "",
                "音频索引": 0,
                "音频开始": 0.0,      
                "图片索引": -1,
                "尾帧位置": -1,
                "转场": 1
            }

            if isinstance(line, list):
                try:
                    if len(line) >= 1:
                        line_data["音频时长"] = float(line[0]) if line[0] else 15.0
                    if len(line) >= 2:
                        line_data["提示词文本"] = line[1] if line[1] else ""
                    if len(line) >= 3:
                        line_data["音频索引"] = int(line[2]) if line[2] else 0
                    if len(line) >= 4:
                        line_data["音频开始"] = float(line[3]) if line[3] else 0.0
                    if len(line) >= 5:
                        line_data["图片索引"] = int(line[4]) if line[4] else -1
                    if len(line) >= 6:
                        line_data["尾帧位置"] = int(line[5]) if line[5] else -1
                    if len(line) >= 7:
                        line_data["转场"] = int(line[6]) if line[6] is not None else 1
                except:
                    pass
            elif isinstance(line, str):
                line_data["提示词文本"] = line

            scene_data.append(line_data)
            segment_durations.append(line_data["音频时长"])

        return (total_count, scene_data, segment_durations)