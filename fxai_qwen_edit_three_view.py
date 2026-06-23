import torch
import re
import math
import comfy.utils
import node_helpers
import comfy.model_management
import sys
import os

from fxai_controlnet import easyControlnet

class FxAiQwenEditThreeView:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 960, "min": 512, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1280, "min": 512, "max": 4096, "step": 8}),
                "用户提示词": ("STRING", {"forceInput": True}),
                "人物头像": ("IMAGE",),
            },
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

    def encode(self, clip, vae, width, height, 用户提示词, 人物头像,):
	
        sys_prompt = "Give a detailed description of the character's facial features, face shape and facial characteristics in the picture, discard other elements including the background and hairstyle, and replace the original background with white. Afterwards, strictly replicate the character's original facial features, face shape and facial characteristics as the main body in accordance with the user's requirements for outfit and hairstyle, and generate new full-body three-view images on a white background in sequence: front view, left side view and back view."

        target_latent_h = self.align_size_8(height) // 8
        target_latent_w = self.align_size_8(width) // 8

        img_batch = self.to_batch(人物头像)
        if isinstance(img_batch, list):
            avatar_img = img_batch[0][:1]
        else:
            avatar_img = img_batch[:1]

        image_part = "图1: <|vision_start|><|image_pad|><|vision_end|>Generate full-body three-view images with a pure white background strictly based on the facial features, face shape, facial characteristics, outfit and hairstyle requirements of the person in the provided picture, arranged in the order of front view, left side view and rear view.\nRequirements for the character's outfit and hairstyle:\n"

        images_vl, ref_latents = [], []
        samples = avatar_img.movedim(-1, 1)
        total = int(384 * 384)

        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)

        s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
        images_vl.append(s.movedim(1, -1))

        total = int(1024 * 1024)
        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = round(samples.shape[3] * scale_by / 8.0) * 8
        height = round(samples.shape[2] * scale_by / 8.0) * 8

        s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
        ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

        template = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n"
        tokens = clip.tokenize(image_part + 用户提示词, vision_images=images_vl, llama_template=template)
        positive = clip.encode_from_tokens_scheduled(tokens)
        positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)

        neg_raw = "丑陋，模糊，低分辨率，最差质量，低质量，JPEG伪影，解剖结构错误，畸形，毁容，突变，多余肢体，多余手臂，多余腿，畸形肢体，手部画得差，手部畸形，多余手指，缺少手指，手指缺失，手指融合，脸部画得差，脸部畸形，毁容的脸，斗鸡眼，长脖子，多余的眼睛，文字，词语，签名，水印，用户名，标志，边框，画框，平铺重复，画得差，出框，错误，画面裁切，畸形的身体"
        neg_tokens = clip.tokenize(neg_raw)
        negative = clip.encode_from_tokens_scheduled(neg_tokens)

        device = comfy.model_management.intermediate_device()
        latent = torch.zeros(1, 4, target_latent_h, target_latent_w, device=device, dtype=torch.float16)
        latent_out = {"samples": latent}

        return (positive, negative, latent_out)