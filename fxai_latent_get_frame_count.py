import torch

class FxAiLatentGetFrameCount:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "潜空间序列": ("LATENT",),
            },
        }

    RETURN_TYPES = ("INT",)  # 输出：总帧数（数字）
    RETURN_NAMES = ("总帧数",)
    FUNCTION = "run"
    CATEGORY = "凤希AI/工具"

    def run(self, 潜空间序列):
        # 从潜空间中取出样本数据
        samples = 潜空间序列["samples"]
        
        # 获取帧数：shape[1] 就是 ComfyUI 潜空间的帧数维度
        总帧数 = samples.shape[1]
        
        # 返回数字
        return (总帧数,)