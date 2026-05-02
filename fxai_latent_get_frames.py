import torch

class FxAiLatentGetFrames:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "潜空间序列": ("LATENT",),
                "起始索引": ("INT", {"default": -1, "min": -1, "max": 9999}),
                "长度": ("INT", {"default": 1, "min": -9999, "max": 9999}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "凤希AI/工具"

    def run(self, 潜空间序列, 起始索引, 长度):
        samples = 潜空间序列["samples"].clone()
        总帧数 = samples.shape[1]

        # 索引 -1 自动变成最后一帧
        if 起始索引 == -1:
            起始索引 = 总帧数 - 1

        # 安全范围限制
        起始索引 = max(0, min(起始索引, 总帧数 - 1))

        # ==========================================
        # 核心逻辑：长度正→向后取，负→向前取
        # ==========================================
        if 长度 > 0:
            结束索引 = 起始索引 + 长度
            结束索引 = min(结束索引, 总帧数)
        else:
            结束索引 = 起始索引 + 1
            起始索引 = 起始索引 + 长度
            起始索引 = max(起始索引, 0)

        # 切片取出（保持 5 维原样）
        选中帧 = samples[:, 起始索引:结束索引]

        return ({"samples": 选中帧},)