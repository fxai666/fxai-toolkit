# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权

from fxai_audio_utils import (
    resolve_audio_path,
    load_audio_tensor_from_file,
)


class FxAiAudioSelector:
    """选择音频并输出 AUDIO（选择按钮走前端弹窗）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "音频文件": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频",)
    FUNCTION = "select"
    CATEGORY = "凤希AI/音频"

    @classmethod
    def VALIDATE_INPUTS(cls, 音频文件=""):
        try:
            if 音频文件:
                resolve_audio_path(音频文件)
            return True
        except Exception as e:
            return str(e)

    def select(self, 音频文件=""):
        print(f"✅ [凤希AI] 已加载音频：{音频文件}，即将进行人声分离")
        return (load_audio_tensor_from_file(音频文件),)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
