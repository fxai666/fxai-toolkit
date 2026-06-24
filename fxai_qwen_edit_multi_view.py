import torch
import re
import math
import comfy.utils
import node_helpers
import comfy.model_management
import sys
import os
from fxai_controlnet import easyControlnet
import cv2  # 新增：用于图像背景分割
import numpy as np

class FxAiQwenEditMultiView:
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
    
    # 新增：人脸分割+背景移除，只保留纯人脸区域（白色背景）
    def remove_avatar_background(self, avatar_img):
        # 将tensor转为cv2可处理的格式 (H,W,C) 0-255
        img_np = (avatar_img.squeeze().cpu().numpy() * 255).astype(np.uint8)
        if img_np.shape[-1] == 4:
            img_np = img_np[:, :, :3]  # 去掉alpha通道
        
        # 使用cv2人脸检测（Haar级联）定位人脸区域
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # 创建纯白背景画布
        white_bg = np.ones_like(img_np) * 255
        
        if len(faces) > 0:
            # 提取最大的人脸区域（避免多脸干扰）
            x, y, w, h = max(faces, key=lambda rect: rect[2]*rect[3])
            # 扩大人脸框，确保完整包含面部特征
            expand_ratio = 0.2
            x_exp = max(0, int(x - w*expand_ratio))
            y_exp = max(0, int(y - h*expand_ratio))
            w_exp = min(img_np.shape[1] - x_exp, int(w * (1 + 2*expand_ratio)))
            h_exp = min(img_np.shape[0] - y_exp, int(h * (1 + 2*expand_ratio)))
            
            # 把人脸区域复制到纯白背景上
            white_bg[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp] = img_np[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp]
        
        # 转回tensor格式 (1,H,W,C) 0-1
        processed_img = torch.from_numpy(white_bg.astype(np.float32) / 255).unsqueeze(0)
        return processed_img

    def encode(self, clip, vae, width, height, 用户提示词, 人物头像):
        target_latent_h = self.align_size_8(height) // 8
        target_latent_w = self.align_size_8(width) // 8

        img_batch = self.to_batch(人物头像)
        if isinstance(img_batch, list):
            avatar_img = img_batch[0][:1]
        else:
            avatar_img = img_batch[:1]
        
        # ========== 关键修改1：预处理头像，强制移除背景，只保留人脸+纯白背景 ==========
        avatar_img = self.remove_avatar_background(avatar_img)

        # ===================== 重写正向视觉提示词（强化背景约束） =====================
        image_part = """Picture 1: <|vision_start|><|image_pad|><|vision_end|>
The reference picture is only a close-up avatar of a person with a pure white background, only facial features, face shape, bone structure and skin tone are valid reference information; 
ALL other elements in the reference image (including background, clothing, hairstyle, accessories, body parts beyond the face) are INVALID and must be completely ignored.
Generate a standard multi-view character sheet with studio-level rendering. The whole canvas has a pure white seamless studio background (non-negotiable), soft and uniform diffused lighting, no debris, no messy environmental elements, 85mm portrait lens, realistic skin texture, ultra-high definition details, natural soft light and shadow, commercial portrait photography texture.

Fixed canvas layout rules (strict left-to-right layout, 4 independent display areas):
1. Left area: 25% of the total canvas width, close-up half-body portrait of the character's front face. Must retain 100% of the original facial identity, facial bones, facial proportions, eye/nose/mouth features and skin tone from the avatar reference image; hairstyle and clothes here follow user custom requirements, not limited to the reference avatar.
2. Right area: occupies 75% of the canvas width, vertically divided into three equal full-body display blocks from top to bottom, fixed order:
   Block 1: Full body front view of the character
   Block 2: Full body left side view of the character
   Block 3: Full body back view of the character

Core mandatory identity rules (only facial features are locked):
1. All 4 drawing areas must belong to the same person, facial identity, facial bone structure, facial proportions, eye shape, nose shape, lip shape, skin tone and facial contour are completely consistent with the reference avatar, no facial distortion, no face swapping;
2. Hairstyle, hair color, clothing style, clothing details, body shape, height, body proportions, accessories and shoes are completely controlled by user prompt, do not need to be consistent with the reference avatar (the reference image only has a headshot without complete hair and clothes);
3. All full-body views must be complete from head to toe, no missing limbs, no body cropping, no distorted limbs;
4. Uniform light intensity and light direction for all four areas, PURE WHITE BACKGROUND (no text, watermarks, color blocks, extra objects, or any elements from the reference image background);
5. The facial recognition features of the character remain unchanged in all perspectives, even if the hairstyle and outfit are replaced.

User custom creation requirements:\n"""

        images_vl, ref_latents = [], []
        samples = avatar_img.movedim(-1, 1)

        total_pixel = samples.shape[3] * samples.shape[2]
        scale_by_384 = 384 / max(samples.shape[2], samples.shape[3])
        width_384 = self.align_size_8(round(samples.shape[3] * scale_by_384))
        height_384 = self.align_size_8(round(samples.shape[2] * scale_by_384))
        s = comfy.utils.common_upscale(samples, width_384, height_384, "bicubic", "disabled")
        images_vl.append(s.movedim(1, -1))

        scale_by_1024 = 1024 / max(samples.shape[2], samples.shape[3])
        width_1024 = self.align_size_8(round(samples.shape[3] * scale_by_1024))
        height_1024 = self.align_size_8(round(samples.shape[2] * scale_by_1024))
        s_high = comfy.utils.common_upscale(samples, width_1024, height_1024, "bicubic", "disabled")
        ref_latent = vae.encode(s_high.movedim(1, -1)[:, :, :, :3].contiguous())
        ref_latents.append(ref_latent)

        template = """<|im_start|>system\n1. Extract only the core facial features from the input avatar reference image: face shape, cheekbone/jawbone skeletal structure, eye/nose/lip shape, facial feature spacing, skin tone, facial contour; 
IGNORE ALL other elements in the reference image (including background, clothing, hairstyle, body parts beyond the face, any objects in the background).
2. Generate multi-view character sheet strictly according to the fixed layout: left half-body close-up + right three vertical full-body views (front / left side / back).
3. Hard lock rule: The facial identity of the character in all views must be exactly the same as the reference avatar, cannot change facial features, face shape and bone structure.
4. Free custom rule: Hairstyle, hair length, clothing, body shape, height, accessories are fully subject to user prompt content, no need to match the reference avatar.
5. Mandatory background rule: All generated images must have a pure white seamless studio background, no elements from the reference image background are allowed to appear.
6. Keep high-definition realistic skin texture, uniform soft studio lighting, pure white background, complete full body without cropping, no facial distortion or cross-face.
<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"""

        tokens = clip.tokenize(image_part + 用户提示词, vision_images=images_vl, llama_template=template)
        positive = clip.encode_from_tokens_scheduled(tokens)

        neg_raw = """facial distortion, face swap, different person, inconsistent facial features, deformed facial bones, disproportionate facial features, blurred face, missing facial details, limbs missing, body cropped, twisted body, 
background elements from reference image, non-white background, messy background, text, watermark, color stain, multi-person, low resolution, pixelated, uneven lighting, facial contour change, eye/nose/mouth shape mutation,
any objects from reference image background, clothing from reference image, hairstyle from reference image, accessories from reference image, body parts beyond face from reference image"""
        neg_tokens = clip.tokenize(neg_raw)
        negative = clip.encode_from_tokens_scheduled(neg_tokens)

        positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)

        device = comfy.model_management.intermediate_device()
        target_latent_h = self.align_size_8(height) // 8
        target_latent_w = self.align_size_8(width) // 8
        latent = torch.zeros(1, 4, target_latent_h, target_latent_w, device=device, dtype=torch.float16)
        latent_out = {"samples": latent}

        return (positive, negative, latent_out)