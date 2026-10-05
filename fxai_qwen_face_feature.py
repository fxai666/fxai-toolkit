import torch
import math
import comfy.utils
import node_helpers
import comfy.model_management

DEFAULT_NEGATIVE = "丑陋，模糊，低分辨率，最差质量，低质量，JPEG伪影，解剖结构错误，畸形，毁容，突变，多余肢体，多余手臂，多余腿，畸形肢体，手部画得差，手部畸形，多余手指，缺少手指，手指缺失，手指融合，脸部画得差，脸部畸形，毁容的脸，斗鸡眼，长脖子，多余的眼睛，边框，画框，平铺重复，画得差，出框，错误，画面裁切，畸形的身体，杂乱背景，过度滤镜，画面暗沉发黑"

# image1 = 参考头像，image2 = 人物图片
POSITIVE_WITH_AVATAR = "保持<image1>人物的面部特征不变，给<image1>穿上<image2>的衣服并保持与<image2>一致的发型及饰品，生成的正面头部特写图，纯白色背景。"

# 无参考头像：仅人物图片，反生成人物面部特征图
POSITIVE_NO_AVATAR = "根据<image1>中人物的形象，生成该人物的正面头部特写人物面部特征图，面部特征与<image1>保持一致，纯白色背景。"


def _prepare_image(img, 宽度):
    samples = img[:1].movedim(-1, 1)
    ratio = samples.shape[3] / samples.shape[2]

    target_w = round(math.sqrt(宽度 * 宽度 * ratio) / 32) * 32
    target_h = round(math.sqrt(宽度 * 宽度 / ratio) / 32) * 32
    target_w, target_h = max(32, target_w), max(32, target_h)

    if (target_w, target_h) == (samples.shape[3], samples.shape[2]):
        s = img[:1]
    else:
        s = comfy.utils.common_upscale(samples, target_w, target_h, "lanczos", "disabled").movedim(1, -1)

    rgb = s[:, :, :, :3]
    if s.shape[-1] > 3:
        rgb = rgb * s[:, :, :, 3:] + (1.0 - s[:, :, :, 3:])
    return rgb, s


def _to_image_list(img):
    if isinstance(img, list):
        return img
    return [img[i:i+1] for i in range(img.shape[0])]


class FxAiQwenFaceFeature:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "宽度": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 32}),
                "高度": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 32}),
            },
            "optional": {
                "人物图片": ("IMAGE",),
                "参考头像": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("正向条件", "负向条件", "潜空间")
    FUNCTION = "encode"
    CATEGORY = "凤希AI/图片"

    def encode(self, clip, vae, 宽度, 高度, 人物图片=None, 参考头像=None):
        负向提示词 = DEFAULT_NEGATIVE

        # 有参考头像：image1=头像，image2=人物图片；无头像：image1=人物图片
        if 参考头像 is not None:
            正向提示词 = POSITIVE_WITH_AVATAR
            ordered = []
            for img in _to_image_list(参考头像):
                if img is not None:
                    ordered.append(img)
            if 人物图片 is not None:
                for img in _to_image_list(人物图片):
                    if img is not None:
                        ordered.append(img)
        else:
            正向提示词 = POSITIVE_NO_AVATAR
            ordered = []
            if 人物图片 is not None:
                for img in _to_image_list(人物图片):
                    if img is not None:
                        ordered.append(img)

        ref_latents = []
        images_vl = []

        for img in ordered:
            rgb, s = _prepare_image(img, 宽度)
            images_vl.append(rgb)
            ref_latents.append(vae.encode(s))

        keep_vision = len(ref_latents) == 0
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(正向提示词, images=images_vl, keep_vision=keep_vision, prevent_empty_text=True))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(负向提示词, images=images_vl, keep_vision=keep_vision, prevent_empty_text=True))

        if len(ref_latents) > 0:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)

        latent = torch.zeros([1, 64, 高度 // 16, 宽度 // 16], device=comfy.model_management.intermediate_device())
        return (positive, negative, {"samples": latent})
