from comfy_extras.nodes_lt import get_noise_mask, LTXVAddGuide, _append_guide_attention_entry
import comfy
import torch
from fxai_image_utils import ImageSizeController

class FxAiLtxvGuideFrames:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "正向条件": ("CONDITIONING",),
                "负向条件": ("CONDITIONING",),
                "视频VAE": ("VAE",),
                "视频潜变量": ("LATENT",),
                "引导图批量": ("IMAGE",),
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
        vae_device = 视频VAE.device

        # latent shape: [B, C, T, H, W]
        b, c, latent_length, latent_h, latent_w = latent_image.shape
        # LTXV缩放取单数值（时间轴缩放忽略，只用空间缩放）
        space_scale = scale_factors[1] if isinstance(scale_factors, tuple) else scale_factors
        target_img_w = latent_w * space_scale
        target_img_h = latent_h * space_scale
        print(f"{target_img_w},{target_img_h}")

        img_ctrl = ImageSizeController(canvas_w=int(target_img_w), canvas_h=int(target_img_h), bg_color=(255,255,255))

        # 统一转为单帧[1,H,W,C]列表
        if isinstance(引导图批量, list):
            frame_list = []
            for f in 引导图批量:
                if f.dim() == 3:
                    frame_list.append(f.unsqueeze(0))
                elif f.dim() == 4:
                    frame_list.append(f)
        else:
            frame_list = [引导图批量[i:i+1] for i in range(引导图批量.shape[0])]

        if not frame_list:
            return (正向条件, 负向条件, {"samples": latent_image, "noise_mask": noise_mask})

        for frame in frame_list:
            resized_img = img_ctrl.crop_fill_to_canvas(frame)
            resized_img = resized_img.to(vae_device)

            image_1, t = LTXVAddGuide.encode(视频VAE, latent_w, latent_h, resized_img, scale_factors)

            frame_idx, latent_idx = LTXVAddGuide.get_latent_index(正向条件, latent_length, len(image_1), 指定帧索引, scale_factors)
            assert latent_idx + t.shape[2] <= latent_length, "引导帧超出视频长度范围"

            正向条件, 负向条件, latent_image, noise_mask = LTXVAddGuide.append_keyframe(
                正向条件, 负向条件, frame_idx, latent_image, noise_mask, t, 引导强度, scale_factors
            )

            pre_filter_count = t.shape[2] * t.shape[3] * t.shape[4]
            guide_latent_shape = list(t.shape[2:])
            正向条件, 负向条件 = _append_guide_attention_entry(正向条件, 负向条件, pre_filter_count, guide_latent_shape, strength=引导强度)

        return (正向条件, 负向条件, {"samples": latent_image, "noise_mask": noise_mask})