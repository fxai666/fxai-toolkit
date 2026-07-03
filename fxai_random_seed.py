import random
import time

class FxAiRandomSeed:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {}
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("随机种",)
    FUNCTION = "generate"
    CATEGORY = "凤希AI/工具"

    def generate(self):
        随机数值 = random.randint(0, 1125899906842624)
        return (随机数值,)

    @classmethod
    def IS_CHANGED(s):
        return str(time.time())