import json
from typing import List, Dict, Any


_TRUE_WORDS = {"1", "true", "yes", "y", "on", "开", "开启", "是", "要", "续接"}


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in _TRUE_WORDS
    return bool(v)


class FxAiScreenManager:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lines_data": ("STRING", {"multiline": False, "default": "[]"}),
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
                # 每个场景自己决定要不要接上一段的续帧
                "场景续接": False,
            }

            if isinstance(line, list):
                try:
                    if len(line) >= 1:
                        line_data["时长"] = float(line[0])
                    if len(line) >= 2:
                        line_data["台词"] = line[1]
                    if len(line) >= 3:
                        line_data["素材"] = line[2]
                    # 第 4 项 = 本场景的续接勾选（老数据只有 3 项，缺省关闭）
                    if len(line) >= 4:
                        line_data["场景续接"] = _to_bool(line[3])
                except Exception:
                    pass
            elif isinstance(line, dict):
                try:
                    line_data["时长"] = float(line.get("时长", line.get("音频时长", 15.0)))
                    line_data["台词"] = line.get("台词", line.get("提示词文本", ""))
                    line_data["素材"] = line.get("素材", line.get("图片地址", line.get("图片文件", "")))
                    for k in ("场景续接", "启用场景续接", "续接", "scene_relay", "relay"):
                        if k in line:
                            line_data["场景续接"] = _to_bool(line[k])
                            break
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