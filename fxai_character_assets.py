class FxAiCharacterAssets:
    @classmethod
    def INPUT_TYPES(cls):
        # 空输入 → 没有目录、没有序号、没有任何控件
        return {}

    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "凤希AI/角色"

    def run(self):
        return ()