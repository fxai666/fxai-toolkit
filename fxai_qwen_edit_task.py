import torch
import re
import math
import comfy.utils
import node_helpers
import comfy.model_management

DEFAULT_SYS = "详细描述输入的每张图片的信息（人物、色彩、造型、尺寸、纹理、物件、背景等），随后强调严格遵守用户的文字要求进行图片的调整编辑，最终生成一张完全满足用户要求的新图；执行要求：先逐项拆解人物特征、原图颜色、外形轮廓、画幅大小、表面肌理、画面包含物体、环境背景各大要素并完整记述，接着逐条对照用户文字要求，列明修改点与改动逻辑，根据用户要求严格引用原图的内容，没有要求的直接抛弃，最后根据尺寸自动补全画面构图产出全新图像。"

class FxAiQwenEditEnhancedV2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 960, "min": 512, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1280, "min": 512, "max": 4096, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
            },
            "optional": {
                "用户提示词": ("STRING", {"forceInput": True}),
                "负面提示词": ("STRING", {"forceInput": True}),
                "图片列表": ("IMAGE",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "凤希AI/提示词"

    def to_batch(self, img):
        if img is None:
            return None
        if isinstance(img, list):
            return img
        return img

    def encode(self, clip, vae, width, height, batch_size, 用户提示词="", 负面提示词=None, 图片列表=None, unique_id=None):
        user_text = 用户提示词.replace("\n", " ").strip()
        target_latent_h, target_latent_w = height // 8, width // 8

        if not isinstance(图片列表, list):
            if isinstance(图片列表, torch.Tensor) and 图片列表.dim() == 4:
                img_batch = [图片列表]
            else:
                img_batch = []
        else:
            img_batch = [t for t in 图片列表 if isinstance(t, torch.Tensor) and t.dim() == 4]

        img_count = len(img_batch)
        image_tags = []
        final_image_sequence = []

        for i in range(img_count):
            image_tags.append(f"图{i+1}: <|vision_start|><|image_pad|><|vision_end|>")
            final_image_sequence.append(img_batch[i])

        image_part = "".join(image_tags)
        user_final = f" {image_part} {user_text}".strip()

        images_vl, ref_latents = [], []
        # 逐图预处理视觉输入与参考潜变量
        for img in final_image_sequence:
            one_img = img
            samples = one_img.movedim(-1,1).contiguous()
            _, _, h, w = samples.shape

            # Qwen VL标准384基准缩放
            scale = math.sqrt(384 * 384 / (w * h))
            scaled = comfy.utils.common_upscale(samples, round(w * scale), round(h * scale), "area", "disabled")
            images_vl.append(scaled.movedim(1, -1))

            # 生成参考latent，尺寸对齐8倍数
            if vae is not None:
                scale2 = math.sqrt(1024 * 1024 / (w * h))
                w2 = round(w * scale2 / 8) * 8
                h2 = round(h * scale2 / 8) * 8
                s2 = comfy.utils.common_upscale(samples, w2, h2, "area", "disabled")
                ref_latents.append(vae.encode(s2.movedim(1, -1)[:, :, :, :3]))

        # 组装Qwen对话模板
        template = f"<|im_start|>system\n{DEFAULT_SYS}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n<|im_end|>"

        # 正向编码
        tokens = clip.tokenize(user_final, vision_images=images_vl, llama_template=template)
        positive = clip.encode_from_tokens_scheduled(tokens)

        # 负面提示词兜底
        if not 负面提示词:
            负面提示词 = "丑陋，模糊，低分辨率，最差质量，低质量，JPEG伪影，解剖结构错误，畸形，毁容，突变，多余肢体，多余手臂，多余腿，畸形肢体，手部画得差，手部畸形，多余手指，缺少手指，手指缺失，手指融合，脸部画得差，脸部畸形，毁容的脸，斗鸡眼，长脖子，多余的眼睛，文字，词语，签名，水印，用户名，标志，边框，画框，平铺重复，画得差，出框，错误，画面裁切，畸形的身体"
        neg_tokens = clip.tokenize(负面提示词.strip())
        negative = clip.encode_from_tokens_scheduled(neg_tokens)

        if vae is not None and ref_latents:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)

        # 初始化latent
        latent = torch.zeros(1, 4, target_latent_h, target_latent_w, device=comfy.model_management.intermediate_device())
        if vae is not None and final_image_sequence:
            first_img = final_image_sequence[0]
            base = first_img.movedim(-1, 1)
            resized = comfy.utils.common_upscale(base, target_latent_w * 8, target_latent_h * 8, "lanczos", "center")
            latent = vae.encode(resized.movedim(1, -1)[:, :, :, :3])

        # batch扩批优化，避免无意义repeat
        if batch_size > 1:
            positive *= batch_size
            negative *= batch_size
            latent_samples = latent.repeat(batch_size, 1, 1, 1)
        else:
            latent_samples = latent

        latent_out = {"samples": latent_samples}

        return (positive, negative, latent_out)