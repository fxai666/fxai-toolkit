import torch
import math
import comfy.utils
import node_helpers
import comfy.model_management

DEFAULT_SYS = "详细描述输入的每张图片的信息（人物、色彩、造型、尺寸、纹理、物件、背景等），随后强调严格遵守用户的文字要求进行图片的调整编辑，最终生成一张完全满足用户要求的新图；执行要求：先逐项拆解人物特征、原图颜色、外形轮廓、画幅大小、表面肌理、画面包含物体、环境背景各大要素并完整记述，接着逐条对照用户文字要求，列明修改点与改动逻辑，根据用户要求严格引用原图的内容，没有要求的直接抛弃，最后根据尺寸自动补全画面构图产出全新图像。"

class FxAiQwenEditEnhanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "用户提示词": ("STRING", {"forceInput": True}),
                "负面提示词": ("STRING", {"forceInput": True}),
                "width": ("INT", {"default": 960, "min": 512, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1280, "min": 512, "max": 4096, "step": 8}),
            },
            "optional": {
                "vae": ("VAE",),
                "人物列表": ("IMAGE",),
                "图片列表": ("IMAGE",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "凤希AI/图片"

    def to_batch(self, img):
        if img is None:
            return None
        if isinstance(img, list):
            return img
        return img

    def encode(self, clip, 用户提示词, 负面提示词, width, height, vae=None, 人物列表=None, 图片列表=None, unique_id=None):
        user_text = 用户提示词.replace("\n", " ").strip()
        target_latent_h, target_latent_w = height // 8, width // 8

        per_batch = self.to_batch(人物列表)
        img_batch = self.to_batch(图片列表)

        # 统计数量
        per_count = 0
        if per_batch is not None:
            per_count = len(per_batch) if isinstance(per_batch, list) else per_batch.shape[0]

        img_count = 0
        if img_batch is not None:
            img_count = len(img_batch) if isinstance(img_batch, list) else img_batch.shape[0]

        image_part = ""
        final_image_sequence = []

        for idx in range(per_count):
            num = idx + 1
            tag = f"人物{num}"
            image_part += f"{tag}: <|vision_start|><|image_pad|><|vision_end|>"

            if isinstance(per_batch, list):
                cur_img = per_batch[idx]
            else:
                cur_img = per_batch[idx:idx+1]

            final_image_sequence.append(cur_img)

        for idx in range(img_count):
            num = idx + 1
            tag = f"图{num}"
            image_part += f"{tag}: <|vision_start|><|image_pad|><|vision_end|>"

            if isinstance(img_batch, list):
                cur_img = img_batch[idx]
            else:
                cur_img = img_batch[idx:idx+1]

            final_image_sequence.append(cur_img)

        user_final_prompt = (image_part + " " + user_text).strip()

        images_vl = []
        ref_latents = []
        for img in final_image_sequence:
            samples = img.movedim(-1, 1).contiguous()
            _, _, h, w = samples.shape

            # Qwen视觉输入缩放 384基准
            scale_vision = math.sqrt(384 * 384 / (w * h))
            vis_scaled = comfy.utils.common_upscale(samples, round(w * scale_vision), round(h * scale_vision), "area", "disabled")
            images_vl.append(vis_scaled.movedim(1, -1))

            if vae is not None:
                # 参考latent 1024基准
                scale_ref = math.sqrt(1024 * 1024 / (w * h))
                w_ref = round(w * scale_ref / 8) * 8
                h_ref = round(h * scale_ref / 8) * 8
                ref_scaled = comfy.utils.common_upscale(samples, w_ref, h_ref, "area", "disabled")
                lat = vae.encode(ref_scaled.movedim(1, -1)[:, :, :, :3])
                ref_latents.append(lat)

        # Qwen对话模板
        chat_template = f"<|im_start|>system\n{DEFAULT_SYS}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n<|im_end|>"
        tokens = clip.tokenize(user_final_prompt, vision_images=images_vl, llama_template=chat_template)
        positive = clip.encode_from_tokens_scheduled(tokens)

        # 负面提示词兜底
        neg_input = 负面提示词.strip()
        if not neg_input:
            neg_input = "丑陋，模糊，低分辨率，最差质量，低质量，JPEG伪影，解剖结构错误，畸形，毁容，突变，多余肢体，多余手臂，多余腿，畸形肢体，手部画得差，手部畸形，多余手指，缺少手指，手指缺失，手指融合，脸部画得差，脸部畸形，毁容的脸，斗鸡眼，长脖子，多余的眼睛，文字，词语，签名，水印，用户名，标志，边框，画框，平铺重复，画得差，出框，错误，画面裁切，畸形的身体"
        neg_tokens = clip.tokenize(neg_input)
        negative = clip.encode_from_tokens_scheduled(neg_tokens)

        if vae is not None and ref_latents:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)

        dev = comfy.model_management.intermediate_device()
        latent_base = torch.zeros(1, 4, target_latent_h, target_latent_w, device=dev)

        latent_output = {"samples": latent_base}
        return (positive, negative, latent_output)