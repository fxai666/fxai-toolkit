import torch
import re
import math
import comfy.utils
import node_helpers
import comfy.model_management
import sys
import os

from fxai_controlnet import easyControlnet

DEFAULT_SYS = """生成纯白色背景的新图像，OpenPose姿态图5个位置说明：第1个位置放人脸特写视图、第2个位置放正面视图、第3个位置放左侧面视图、第4个位置放背面视图、第5个位置放右侧面视图。"""

class FxAiQwenEditMultiView:
    VIEW_CONFIGS = [
        "人脸特写视图，<sks> close-up face, detailed facial portrait, only face,front view, same exact face as reference avatar",
        "正面视图,<sks> front view, full body, facing forward, identical facial features",
        "左侧面视图,<sks> left side profile, full body left view, same person face",
        "背面视图,<sks> back view, rear full body, face consistent with avatar front",
        "右侧面视图,<sks> right side profile, full body right view, same facial structure",
    ]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 960, "min": 512, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1280, "min": 512, "max": 4096, "step": 8}),
                "用户提示词": ("STRING", {"forceInput": True}),
                "姿势参考图": ("IMAGE",),
                "人物头像": ("IMAGE",),
            },
            "optional": {
                "负面提示词": ("STRING", {"forceInput": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "凤希AI/图片"

    def align_size_8(self, val):
        return math.ceil(val / 8) * 8

    def to_batch(self, img):
        if img is None:
            return None
        if isinstance(img, list):
            return img
        return img

    def encode(self, clip, vae, width, height, 用户提示词, 姿势参考图, 人物头像, 负面提示词=None,unique_id=None):
        user_text = 用户提示词.replace("\n", " ").strip()
        identity_prompt = "【核心身份锁定】全程使用下方人脸身份参考图的人物五官、脸型、面部特征，全程不更换人物、不改变五官，所有视图为同一个人；"
        view_desc = "\n".join(self.VIEW_CONFIGS)
        user_text = (identity_prompt + user_text + "\n\nOpenPose姿势图说明:" + view_desc).strip()

        target_latent_h = self.align_size_8(height) // 8
        target_latent_w = self.align_size_8(width) // 8

        img_batch = self.to_batch(人物头像)
        if isinstance(img_batch, list):
            avatar_img = img_batch[0][:1]
        else:
            avatar_img = img_batch[:1]
        pose_img = 姿势参考图[:1]

        # 明确区分两张图片作用
        image_part = "<|vision_start|><|image_pad|><|vision_end|> <|vision_start|><|image_pad|><|vision_end|>"
        final_image_sequence = [avatar_img, pose_img]
        user_final = (image_part + " " + user_text).strip()

        images_vl, ref_latents = [], []
        for idx, img in enumerate(final_image_sequence):
            one_img = img
            samples = one_img.movedim(-1,1).contiguous()
            _, _, h, w = samples.shape

            scale = math.sqrt(512 * 512 / (w * h))
            new_w = self.align_size_8(round(w * scale))
            new_h = self.align_size_8(round(h * scale))
            scaled = comfy.utils.common_upscale(samples, new_w, new_h, "area", "disabled")
            images_vl.append(scaled.movedim(1, -1))

            # 只把头像编码进reference_latents，姿态图不参与
            if vae is not None and idx == 0:
                scale2 = math.sqrt(1024 * 1024 / (w * h))
                w2 = self.align_size_8(round(w * scale2))
                h2 = self.align_size_8(round(h * scale2))
                s2 = comfy.utils.common_upscale(samples, w2, h2, "area", "disabled")
                lat = vae.encode(s2.movedim(1, -1)[:, :, :, :3])
                ref_latents.append(lat)

        template = f"<|im_start|>system\n{DEFAULT_SYS}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n<|im_end|>"
        tokens = clip.tokenize(user_text, vision_images=images_vl, llama_template=template)
        positive = clip.encode_from_tokens_scheduled(tokens)

        # 负面词增加换脸抑制
        neg_raw = "丑陋，模糊，低分辨率，最差质量，低质量，JPEG伪影，解剖结构错误，畸形，毁容，突变，多余肢体，多余手臂，多余腿，畸形肢体，手部画得差，手部畸形，多余手指，缺少手指，手指缺失，手指融合，脸部画得差，脸部畸形，毁容的脸，斗鸡眼，长脖子，多余的眼睛，文字，词语，签名，水印，用户名，标志，边框，画框，平铺重复，画得差，出框，错误，画面裁切，畸形的身体"
        if 负面提示词 is not None:
            neg_raw = 负面提示词.strip()

        neg_tokens = clip.tokenize(neg_raw)
        negative = clip.encode_from_tokens_scheduled(neg_tokens)

        if vae is not None and len(ref_latents) > 0:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)

        device = comfy.model_management.intermediate_device()
        latent = torch.zeros(1, 4, target_latent_h, target_latent_w, device=device, dtype=torch.float16)
        latent_out = {"samples": latent}

        return (positive, negative, latent_out)