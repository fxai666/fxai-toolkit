import torch

class FxAiLatentClearReplace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "潜空间序列": ("LATENT",),
                "帧索引": ("INT", {"default": 0, "min": -1, "max": 9999}),
            },
            "optional": {
                "新图片": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "凤希AI/工具"

    def run(self, 潜空间序列, 帧索引, 新图片=None):
        # 复制数据
        samples = 潜空间序列["samples"].clone()

        # ==============================================
        # ✅ 精准适配你的 5 维数据: [1, 总帧数, C, H, W]
        # ==============================================
        _, 总帧数, C, H, W = samples.shape

        # 索引处理
        if 帧索引 == -1:
            帧索引 = 总帧数 - 1
        帧索引 = max(0, min(帧索引, 总帧数 - 1))

        # 清空指定帧
        samples[:, 帧索引:帧索引+1] = 0.0

        # 替换图片（不做任何缩放！）
        if 新图片 is not None:
            img = 新图片 * 2.0 - 1.0
            img_latent = img.permute(0, 3, 1, 2).unsqueeze(0)  # 转成 5 维匹配
            samples[:, 帧索引:帧索引+1] = img_latent

        return ({"samples": samples},)