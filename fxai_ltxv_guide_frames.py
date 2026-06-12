from comfy_extras.nodes_lt import get_noise_mask, LTXVAddGuide, _append_guide_attention_entry
import comfy
import torch

class FxAiLtxvGuideFrames:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "正向条件": ("CONDITIONING",),
                "负向条件": ("CONDITIONING",),
                "视频VAE": ("VAE",),
                "视频潜变量": ("LATENT",),
                "引导图批量": ("IMAGE",),  # 图片数组 批量输入
                "指定帧索引": ("INT", {"default": 0, "min": -9999, "max": 9999}),
                "引导强度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("正向条件", "负向条件", "视频潜变量")
    FUNCTION = "process_batch"
    CATEGORY = "凤希AI/LTXV"

    def process_batch(self, 正向条件, 负向条件, 视频VAE, 视频潜变量, 引导图批量, 指定帧索引, 引导强度):
        scale_factors = 视频VAE.downscale_index_formula
        latent_image = 视频潜变量["samples"]
        noise_mask = get_noise_mask(视频潜变量)

        _, _, latent_length, latent_height, latent_width = latent_image.shape

        for img in 引导图批量:
            print("循环取出img shape", img.shape)
            # 新增：剔除多余维度、强制3通道
            while len(img.shape) > 3:
                img = img.squeeze(0)
            if img.shape[-1] == 4:
                img = img[..., :3]
            
            img = img.unsqueeze(0)
            print("循环取出img shape", img.shape)
            image_1, t = LTXVAddGuide.encode(视频VAE, latent_width, latent_height, img, scale_factors)

            frame_idx, latent_idx = LTXVAddGuide.get_latent_index(正向条件, latent_length, len(image_1), 指定帧索引, scale_factors)
            assert latent_idx + t.shape[2] <= latent_length, "引导帧超出视频长度范围"

            正向条件, 负向条件, latent_image, noise_mask = LTXVAddGuide.append_keyframe(
                正向条件, 负向条件, frame_idx, latent_image, noise_mask, t, 引导强度, scale_factors
            )

            pre_filter_count = t.shape[2] * t.shape[3] * t.shape[4]
            guide_latent_shape = list(t.shape[2:])
            正向条件, 负向条件 = _append_guide_attention_entry(正向条件, 负向条件, pre_filter_count, guide_latent_shape, strength=引导强度)

        return (正向条件, 负向条件, {"samples": latent_image, "noise_mask": noise_mask})