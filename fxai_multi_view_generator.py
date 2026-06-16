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

    def make_view_prompts(self, 提示词, 人脸特写, 正面, 右侧面, 背面, 左侧面):
        view_info = [
            (人脸特写, "<sks> close-up face, detailed facial portrait, only face, no upper body,front view"),
            (正面, "<sks> front view, full body, facing forward"),
            (右侧面, "<sks> right side profile, full body right view"),
            (背面, "<sks> back view, rear full body, looking away"),
            (左侧面, "<sks> left side profile, full body left view")
        ]
        active_views = []
        for enable, suffix in view_info:
            if enable:
                active_views.append(suffix)
        
        view_str = ", ".join(active_views)
        full_prompt = f"""1、人物穿搭及发型:：
{提示词}

2、画布排版强制约束：
纯白色纯色背景，无多余杂物，单张画布横向从左到右优先放置人脸特写，再排布其余全身视角[{view_str}]，仅生成勾选的这些视角，禁止额外增加任何人脸/侧面/背面视图；人脸特写仅保留头脸部，无躯干身体，其余视角为完整全身；所有视图人物外貌、发型、穿搭完全一致，标准人物设定三视图稿"""

        sys_prompt = """你是专业人物三视图设定画师，执行规则优先级从上到下，不可颠倒：
1. 【最高强制指令】优先完整替换人物发型、穿搭，严格按照用户给出的穿搭发型描述重塑人物，绝对不能沿用原图自带发型；所有画面（人脸特写+所有三视图）统一使用这套新发型、新服饰，五官轮廓仅参考原图，发型服饰必须更换。
2. 画布规则：纯白纯色空白背景，无杂物、无装饰、无渐变；所有画面横向从左至右依次排列，先放人脸特写，再依次摆放选中的正面、右侧面、背面、左侧面全身视图。
3. 整体为单张完整设定稿，不分割多张图片，构图整洁规整，标准角色设定三视图格式。

禁止违规行为：保留原图旧发型、新增多余人物/视角、彩色/复杂背景、画面拆分多图、特写带身体、不同视图人物造型不统一。"""

        return (full_prompt, sys_prompt)