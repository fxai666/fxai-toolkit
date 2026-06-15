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

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("提示词",)
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
        prompt_list = ["使用提供的人物参考图面部特征按输出要求生成一张白色背景的新图片\n", f"人物穿搭及发型描述(严格遵守)：\n{提示词}\n","生成输出规则：","首先去掉参考图中人物双马尾发型，然后再按要求替换为内容描述中的新发型","接着严格按照要求生成人物视图，生成结果严格按照人物姿势图位置从左到右平铺满画布："]
        for enable, suffix in view_info:
            if enable:
                prompt_list.append(suffix)
        
        batch_prompts = "\n".join(prompt_list)
        return (batch_prompts,)