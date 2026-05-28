import json
from comfy.comfy_types.node_typing import IO
import torch

class FxAiTextPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "任意输入": (IO.ANY, {}),
            }
        }

    # 关键：彻底移除输出端口
    RETURN_TYPES = ()
    FUNCTION = "main"
    OUTPUT_NODE = True  # 仅标记为UI节点，不生成输出

    CATEGORY = "凤希AI/工具"

    def __init__(self):
        self.text_cache = ""

    def main(self, 任意输入=None):
        torch.set_printoptions(edgeitems=6)
        value = "None"
        if isinstance(任意输入, str):
            value = 任意输入
        elif isinstance(任意输入, (int, float, bool)):
            value = str(任意输入)
        elif 任意输入 is not None:
            try:
                value = json.dumps(任意输入, indent=4, ensure_ascii=False)
            except Exception:
                try:
                    value = str(任意输入)
                except Exception:
                    value = "source exists, but could not be serialized."
        torch.set_printoptions()

        self.text_cache = value
        # 关键：只返回ui，不返回result → 无输出端口
        return {"ui": {"text": (value,)}}

    @classmethod
    def get_state(cls, obj):
        return {"text_cache": obj.text_cache}

    @classmethod
    def set_state(cls, obj, state):
        obj.text_cache = state.get("text_cache", "")