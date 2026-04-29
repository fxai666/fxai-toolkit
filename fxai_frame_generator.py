import os
import torch
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

class FxAiFrameGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文件夹路径": ("STRING", {"default": ""}),
                "首帧索引": ("INT", {"default": 0, "min": 0}),
                "尾帧索引": ("INT", {"default": 1, "min": 0}),
                "启用转场": ("BOOLEAN", {"default": True}),
                "输出宽度": ("INT", {"default": 540, "min": 64}),
                "输出高度": ("INT", {"default": 960, "min": 64}),
            },
            "optional": {
                "图片序列": ("IMAGE", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("首帧图", "尾帧图")
    FUNCTION = "generate_frames"
    CATEGORY = "凤希AI/图片"

    # 加载图片
    def load_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            return torch.from_numpy(img_np).unsqueeze(0)
        except:
            return None

    # 封装：居中裁剪（只写一次）
    def crop_center(self, img, target_w, target_h):
        w, h = img.size
        left = (w - target_w) // 2
        top = (h - target_h) // 2
        return img.crop((left, top, left + target_w, top + target_h))

    # ==============================
    # ✅ 最高画质缩放 + 封装精简版
    # ==============================
    def resize_image(self, image_tensor, target_w, target_h):
        if image_tensor is None:
            return None
        
        np_img = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(np_img)
        original_w, original_h = img.size

        if original_w == target_w and original_h == target_h:
            return image_tensor

        if original_w == target_w or original_h == target_h:
            img = self.crop_center(img, target_w, target_h)
        else:
            target_ratio = target_w / target_h
            original_ratio = original_w / original_h

            if original_ratio > target_ratio:
                new_h = target_h
                new_w = int(original_w * new_h / original_h)
            else:
                new_w = target_w
                new_h = int(original_h * new_w / original_w)

            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img = self.crop_center(img, target_w, target_h)

        np_out = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(np_out).unsqueeze(0)

    def generate_frames(self, 文件夹路径, 首帧索引, 尾帧索引, 启用转场, 输出宽度, 输出高度, 图片序列=None):
        image_files = []
        for filename in sorted(os.listdir(文件夹路径)):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image_files.append(os.path.join(文件夹路径, filename))

        total = len(image_files)
        if total == 0:
            return (None, None)

        # 首帧
        if 图片序列 is not None and 图片序列.shape[0] > 0:
            首帧 = 图片序列[-1].unsqueeze(0)
        else:
            首帧 = self.load_image(image_files[首帧索引 % total])

        # 尾帧
        if 启用转场:
            尾帧 = self.load_image(image_files[尾帧索引 % total])
        else:
            尾帧 = 首帧

        if 尾帧 is None:
            尾帧 = 首帧

        # 最高画质处理
        首帧_final = self.resize_image(首帧, 输出宽度, 输出高度)
        尾帧_final = self.resize_image(尾帧, 输出宽度, 输出高度)

        return (首帧_final, 尾帧_final)