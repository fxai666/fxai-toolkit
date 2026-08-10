import json
from typing import List, Dict, Any

class FxAiScreenManager:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lines_data": ("STRING", {"multiline": True, "default": "[]"}),
            }
        }

    RETURN_TYPES = ("INT", "LIST", "LIST")
    RETURN_NAMES = ("总行数", "场景数据", "分段时长")
    FUNCTION = "execute"
    CATEGORY = "凤希AI/影视剧场"

    def execute(self, lines_data="[]") -> tuple[int, List[Dict[str, Any]], List[float]]:
        try:
            if isinstance(lines_data, str):
                lines = json.loads(lines_data.strip())
            elif isinstance(lines_data, (list, dict)):
                lines = lines_data
            else:
                lines = []
        except Exception:
            lines = []

        if not isinstance(lines, list) or len(lines) == 0:
            lines = [[15.0, "", ""]]

        total_count = len(lines)
        scene_data = []
        segment_durations = []

        for idx, line in enumerate(lines):
            line_data = {
                "序号": idx + 1,
                "时长": 15.0,
                "台词": "",
                "素材": "",
            }

            if isinstance(line, list):
                try:
                    if len(line) >= 1:
                        line_data["时长"] = float(line[0])
                    if len(line) >= 2:
                        line_data["台词"] = line[1]
                    if len(line) >= 3:
                        line_data["素材"] = line[2]
                except Exception:
                    pass
            elif isinstance(line, dict):
                try:
                    line_data["时长"] = float(line.get("时长", line.get("音频时长", 15.0)))
                    line_data["台词"] = line.get("台词", line.get("提示词文本", ""))
                    line_data["素材"] = line.get("素材", line.get("图片地址", line.get("图片文件", "")))
                except Exception:
                    pass
            elif isinstance(line, str):
                line_data["台词"] = line

            if isinstance(line_data["素材"], (list, tuple)):
                line_data["素材"] = ",".join(str(x) for x in line_data["素材"] if str(x).strip())
            else:
                line_data["素材"] = str(line_data["素材"] or "")

            scene_data.append(line_data)
            segment_durations.append(line_data["时长"])

        return (total_count, scene_data, segment_durations)
		
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")