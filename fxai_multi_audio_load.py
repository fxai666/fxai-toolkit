import json

class FxAiMultiAudioLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "编号": ("INT", {
                    "forceInput": True,
                    "tooltip": "从1开始，选择要输出的音频编号"
                }),
                "音频一": ("AUDIO", {"forceInput": True}),
            },
            "optional": {
                "音频二": ("AUDIO", {"forceInput": True}),
                "音频三": ("AUDIO", {"forceInput": True}),
                "音频四": ("AUDIO", {"forceInput": True}),
                "音频五": ("AUDIO", {"forceInput": True}),
                "音频六": ("AUDIO", {"forceInput": True}),
                "音频七": ("AUDIO", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("音频", "编号")
    FUNCTION = "execute"
    CATEGORY = "凤希AI/其他"

    def execute(self, 编号, 音频一, 音频二=None, 音频三=None, 音频四=None, 音频五=None, 音频六=None, 音频七=None):
        # 确保编号是有效整数
        try:
            idx = int(编号)
        except (TypeError, ValueError):
            idx = 1
        
        match idx:
            case 1:
                audio = 音频一
            case 2:
                audio = 音频二
            case 3:
                audio = 音频三
            case 4:
                audio = 音频四
            case 5:
                audio = 音频五
            case 6:
                audio = 音频六
            case 7:
                audio = 音频七
            case _:
                audio = 音频一
        
        # 选中的音频为空时，强制使用音频一兜底
        if audio is None:
            audio = 音频一
            
        return (audio, 编号)