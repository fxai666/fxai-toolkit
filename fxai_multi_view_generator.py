class FxAiMultiViewGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING",{"forceInput": True}),
                "人脸特写": ("BOOLEAN", {"default": True}),
                "正面": ("BOOLEAN", {"default": True}),
                "右侧面": ("BOOLEAN", {"default": True}),
                "背面": ("BOOLEAN", {"default": True}),
                "左侧面": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("提示词","系统提示词",)
    FUNCTION = "make_view_prompts"
    CATEGORY = "凤希AI/图片"

    def make_view_prompts(self, 提示词, 人脸特写, 正面, 右侧面, 背面, 左侧面, 采样器输入图片=None):
        view_info = [
            (人脸特写, "人脸特写,close-up face, detailed facial portrait, only face, no upper body,front view"),
            (正面, "正面,front view, full body, facing forward, front character sheet"),
            (右侧面, "右侧面,right side profile, full body right view"),
            (背面, "背面,back view, rear full body, looking away"),
            (左侧面, "左侧面,left side profile, full body left view")
        ]
        prompt_list = ["生成要求：","先根据用户发型要求为人物更换新发型","再严格按照人物穿搭要求根据人物姿态图从左到右生成人脸特写与三视图"]
        for enable, suffix in view_info:
            if enable:
                prompt_list.append(suffix)
        
        batch_prompts = "\n".join(prompt_list)
        return (f"人物穿搭及发型:\n{提示词}","请根据提供的人物参考图面部特征及人物姿态占位图按用户要求生成一张以白色为背景的新图片，必须按用户要求更改人物发型，各人物视图布局之间需要有间隔（禁止人物重叠）。\n"+batch_prompts)