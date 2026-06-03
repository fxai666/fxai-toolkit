import torch
import re
import math
import comfy.utils
import node_helpers
import comfy.model_management

DEFAULT_SYS = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
DEFAULT_NEGATIVE = "worst quality, low quality, ugly, deformed, blurry, watermark, text, signature, username, error, extra fingers, extra limbs"

class FxAiQwenEditEnhanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "用户提示词": ("STRING", {"multiline": True, "dynamicPrompts": True,"placeholder": "使用 人物N / 图N 引用图片"}),
                "负面提示词": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": DEFAULT_NEGATIVE}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
            },
            "optional": {
                "vae": ("VAE",),
                "人物列表": ("IMAGE",),
                "图片列表": ("IMAGE",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STRING", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("positive", "negative", "latent", "final_prompt", "negative_prompt", "first_image", "image_sequence")
    FUNCTION = "encode"
    CATEGORY = "凤希AI/提示词"

    def to_batch(self, img):
        if img is None: return None
        if isinstance(img, list): return torch.cat(img, dim=0) if img else None
        return img

    def encode(self, clip, 用户提示词, 负面提示词, width, height, batch_size, vae=None, 人物列表=None, 图片列表=None, unique_id=None):
        user_text = 用户提示词.replace("\n", " ").strip()
        target_latent_h, target_latent_w = height // 8, width // 8

        per_batch = self.to_batch(人物列表)
        img_batch = self.to_batch(图片列表)
        per_count = per_batch.shape[0] if per_batch is not None else 0
        img_count = img_batch.shape[0] if img_batch is not None else 0

        # ==============================================
        # 【你要的完美逻辑】
        # 1. 正则取出所有图N/人物N
        # 2. 先去重
        # 3. 再循环生成标记 & 对齐图片
        # ==============================================
        pattern = re.compile(r'(人物|图)(\d+)')
        ref_sequence = pattern.findall(user_text)

        # 去重，保留顺序
        unique_refs = []
        seen = set()
        for item in ref_sequence:
            if item not in seen:
                seen.add(item)
                unique_refs.append(item)

        # 生成图片标记部分
        image_part = ""
        final_image_sequence = []

        for (typ, num_str) in unique_refs:
            num = int(num_str)
            img = None

            if typ == "人物" and 1 <= num <= per_count:
                img = per_batch[num-1:num]
            elif typ == "图" and 1 <= num <= img_count:
                img = img_batch[num-1:num]

            if img is not None:
                tag = f"{typ}{num_str}"
                image_part += f"{tag}: <|vision_start|><|image_pad|><|vision_end|>"
                final_image_sequence.append(img)

        # 最终：图片标记 + 完整用户原文（完全不动）
        user_final = (image_part + " " + user_text).strip()

        # 图片批次
        image_batch = torch.cat(final_image_sequence, dim=0) if final_image_sequence else None
        total_img = image_batch.shape[0] if image_batch is not None else 0

        # 图像编码
        images_vl, ref_latents, first_image = [], [], torch.zeros((1,1,1,3))
        if image_batch is not None:
            for i in range(total_img):
                one_img = image_batch[i:i+1]
                samples = one_img.movedim(-1,1).contiguous()
                _, _, h, w = samples.shape

                # VL 编码
                scale = math.sqrt(384*384/(w*h))
                scaled = comfy.utils.common_upscale(samples, round(w*scale), round(h*scale), "area", "disabled")
                images_vl.append(scaled.movedim(1,-1))

                # VAE 编码
                if vae is not None:
                    scale2 = math.sqrt(1024*1024/(w*h))
                    w2 = round(w * scale2 / 8) * 8
                    h2 = round(h * scale2 / 8) * 8
                    s2 = comfy.utils.common_upscale(samples, w2, h2, "area", "disabled")
                    ref_latents.append(vae.encode(s2.movedim(1,-1)[:,:,:,:3]))

                if i == 0:
                    first_image = one_img.clone()

        # 提示词模板
        template = f"<|im_start|>system\n{DEFAULT_SYS}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n<|im_end|>"
        final_prompt = template.format(user_final)

        # 正向
        tokens = clip.tokenize(user_final, vision_images=images_vl, llama_template=template)
        positive = clip.encode_from_tokens_scheduled(tokens)

        # 负向
        neg_tokens = clip.tokenize(负面提示词.strip())
        negative = clip.encode_from_tokens_scheduled(neg_tokens)

        # 参考潜变量
        if vae is not None and ref_latents:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)

        # latent
        if image_batch is not None and vae is not None:
            base = image_batch[0:1].movedim(-1,1)
            resized = comfy.utils.common_upscale(base, target_latent_w*8, target_latent_h*8, "lanczos", "center")
            latent = vae.encode(resized.movedim(1,-1)[:,:,:,:3])
        else:
            latent = torch.zeros(1,4,target_latent_h,target_latent_w, device=comfy.model_management.intermediate_device())

        # batch
        latent_out = {"samples": latent}
        if batch_size > 1:
            positive *= batch_size
            negative *= batch_size
            latent_out["samples"] = latent.repeat(batch_size,1,1,1)

        return (
            positive, negative, latent_out,
            final_prompt, 负面提示词.strip(),
            first_image, image_batch if image_batch is not None else torch.zeros(1,1,1,3)
        )