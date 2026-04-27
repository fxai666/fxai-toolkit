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
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("首帧图", "尾帧图")
    FUNCTION = "generate_frames"
    CATEGORY = "凤希AI/图片"

    # 直接加载完整路径（你要的最简逻辑）
    def load_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            return torch.from_numpy(img_np).unsqueeze(0)
        except:
            return None

    # 缩放不变
    def resize_image(self, image_tensor, target_w, target_h):
        if image_tensor is None:
            return None
        np_img = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(np_img)
        pil_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        np_out = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(np_out).unsqueeze(0)

    def generate_frames(self, 文件夹路径, 首帧索引, 尾帧索引, 启用转场, 输出宽度, 输出高度, 图片序列=None):
        image_files = []
        for filename in sorted(os.listdir(文件夹路径)):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                full_path = os.path.join(文件夹路径, filename)
                image_files.append(full_path)

        total = len(image_files)
        if total == 0:
            return (None, None)

        if 图片序列 is not None and 图片序列.shape[0] > 0:
            首帧 = 图片序列[-1].unsqueeze(0)
        else:
            idx_start = 首帧索引 % total
            首帧 = self.load_image(image_files[idx_start])

        if 启用转场:
            idx_end = 尾帧索引 % total
            尾帧 = self.load_image(image_files[idx_end])
        else:
            尾帧 = 首帧

        # 兜底
        if 尾帧 is None:
            尾帧 = 首帧

        # 缩放
        首帧_final = self.resize_image(首帧, 输出宽度, 输出高度)
        尾帧_final = self.resize_image(尾帧, 输出宽度, 输出高度)

        return (首帧_final, 尾帧_final)