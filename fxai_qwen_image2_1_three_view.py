import torch
import math
import comfy.utils
import node_helpers
import comfy.model_management

DEFAULT_NEGATIVE = "丑陋，模糊，低分辨率，最差质量，低质量，JPEG伪影，解剖结构错误，畸形，毁容，突变，多余肢体，多余手臂，多余腿，畸形肢体，手部画得差，手部畸形，多余手指，缺少手指，手指缺失，手指融合，脸部画得差，脸部畸形，毁容的脸，斗鸡眼，长脖子，多余的眼睛，边框，画框，平铺重复，画得差，出框，错误，画面裁切，畸形的身体，杂乱背景，过度滤镜，画面暗沉发黑"

THREE_VIEW_TEMPLATE = (
    "基于<image1>提供的人物面部特征，完整复刻<image1>人物面部特征生成标准三视图全身角色参考图，纯白背景。\n"
    "按固定顺序输出3个全身视图：\n"
    "1.正面全身\n"
    "2.左侧面全身\n"
    "3.背面全身\n\n"
    "要求：\n"
    "- 三个视图必须是完整的全身照：从头到脚完全在画面内，不能裁切、不能出画、不能截断\n"
    "- 每个视图居中显示，留有适当的头部和脚部空间\n"
    "- 严格保持面部特征、骨骼结构、面部比例一致\n"
    "- 发型和服装在三个视图中完全一致\n"
    "- 纯白背景，无杂物\n\n"
    "用户要求：\n"
)

class FxAiQwenImage21ThreeView:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "正向提示词": ("STRING", {"forceInput": True}),
                "宽度": ("INT", {"default": 960, "min": 512, "max": 4096, "step": 32}),
                "高度": ("INT", {"default": 1280, "min": 512, "max": 4096, "step": 32}),
            },
            "optional": {
                "负向提示词": ("STRING", {"forceInput": True}),
                "图片列表": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("正向条件", "负向条件", "潜空间")
    FUNCTION = "encode"
    CATEGORY = "凤希AI/图片"

    def encode(self, clip, vae, 正向提示词, 宽度, 高度, 负向提示词=None, 图片列表=None):
        if 负向提示词 is None or 负向提示词.strip() == "":
            负向提示词 = DEFAULT_NEGATIVE

        ref_latents = []
        images_vl = []

        if 图片列表 is not None:
            if isinstance(图片列表, list):
                img_list = 图片列表
            else:
                img_list = [图片列表[i:i+1] for i in range(图片列表.shape[0])]

            for img in img_list:
                if img is None:
                    continue

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
                images_vl.append(rgb)

                ref_latents.append(vae.encode(s))

        final_prompt = THREE_VIEW_TEMPLATE + 正向提示词

        keep_vision = len(ref_latents) == 0
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(final_prompt, images=images_vl, keep_vision=keep_vision, prevent_empty_text=True))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(负向提示词, images=images_vl, keep_vision=keep_vision, prevent_empty_text=True))

        if len(ref_latents) > 0:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)

        latent = torch.zeros([1, 64, 高度 // 16, 宽度 // 16], device=comfy.model_management.intermediate_device())
        return (positive, negative, {"samples": latent})
