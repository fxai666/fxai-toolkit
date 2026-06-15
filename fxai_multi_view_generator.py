class FxAiMultiViewGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True, "default": "masterpiece, best quality, full body character, uniform lighting, white background"}),
                "人脸特写": ("BOOLEAN", {"default": True}),
                "正面": ("BOOLEAN", {"default": True}),
                "右侧面": ("BOOLEAN", {"default": True}),
                "背面": ("BOOLEAN", {"default": True}),
                "左侧面": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "采样器输入图片": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",  "LIST", "INT")
    RETURN_NAMES = ("分组正向词集合", "单视角提示词列表", "生成画布总数量")
    FUNCTION = "make_view_prompts"
    CATEGORY = "凤希AI/三视图生成"

    def make_view_prompts(self, 提示词, 人脸特写, 正面, 右侧面, 背面, 左侧面, 采样器输入图片=None):
        view_info = [
            (人脸特写, ", close-up face, head shot, detailed facial portrait"),
            (正面, ", front view, full body, facing forward, front character sheet"),
            (右侧面, ", right side profile, full body right view"),
            (背面, ", back view, rear full body, looking away"),
            (左侧面, ", left side profile, full body left view")
        ]
        prompt_list = ["根据以下要求，严格使用提供的图片人物面貌生成一张白色背景包括人脸特写和三视图的新图片，必须把参考图人物发型换成以下描述的新发型", 提示词,"输出要求：","严格按照人物姿态参考图顺序输出："]
        for enable, suffix in view_info:
            if enable:
                prompt_list.append(suffix)
        
        batch_prompts = "\n".join(prompt_list)
        return (batch_prompts, prompt_list, len(prompt_list))