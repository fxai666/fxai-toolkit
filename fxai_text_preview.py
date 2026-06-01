import json
from comfy.comfy_types.node_typing import IO
import torch

class FxAiTextPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "任意输入": (IO.ANY, {}),
                "cache_text": ("STRING", {"default":""}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "main"
    OUTPUT_NODE = True
    CATEGORY = "凤希AI/工具"

    def main(self, 任意输入=None,cache_text=""):
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
        return {"ui": {"text": (value,)}}