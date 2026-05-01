import torch

class FxAiLatentFrameToImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "潜空间序列": ("LATENT",),
                "VAE": ("VAE",),
                "帧索引": ("INT", {"default": 0, "min": -1, "max": 9999}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "凤希AI/工具"

    def run(self, 潜空间序列, VAE, 帧索引):
        samples = 潜空间序列["samples"].clone()
        
        # 适配你 5维格式：[1, 帧数, C, H, W]
        _, 总帧数, _, _, _ = samples.shape

        # -1 取最后一帧
        if 帧索引 == -1:
            帧索引 = 总帧数 - 1
        # 防越界
        帧索引 = max(0, min(帧索引, 总帧数 - 1))

        # 截取指定单帧 latent，保持维度
        single_latent = samples[:, 帧索引:帧索引+1]

        # VAE 解码成图片
        image = VAE.decode(single_latent)
        # 归一化到 0~1
        image = torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)

        return (image,)