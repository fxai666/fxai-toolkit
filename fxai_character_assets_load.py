import os
import re
import torch
import folder_paths
from fxai_image_utils import (
    load_single_image, fit_to_canvas, grid_concat_images, IMAGE_EXTENSIONS
)

def get_image_dir(subdir=""):
    comfy_root = folder_paths.base_path
    base_dir = "fxai/image"
    target_dir = os.path.join(comfy_root, base_dir)
    
    if subdir:
        subdir = re.sub(r'[\\/*?:"<>|]', "", subdir)
        target_dir = os.path.join(target_dir, subdir)
    
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

class FxAiCharacterAssetsLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片路径列表": ("STRING", {"forceInput": True}),
                "缩小倍数": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "INT")
    RETURN_NAMES = ("图片列表", "遮罩列表", "网格拼接大图", "总数量")
    FUNCTION = "load_images"
    CATEGORY = "凤希AI/角色"

    def load_images(self, 图片路径列表, 缩小倍数):
        path_list = [p.strip() for p in 图片路径列表.split(",") if p.strip()]
        if not path_list:
            return (None, None, None, 0)

        images = []
        masks = []

        for rel_path in path_list:
            parts = rel_path.split("/", 1)
            if len(parts) != 2:
                continue

            subdir, filename = parts
            full_dir = get_image_dir(subdir)
            full_path = os.path.join(full_dir, filename)

            if not os.path.exists(full_path):
                print(f"[凤希] 图片不存在：{full_path}")
                continue

            try:
                # 统一使用工具函数加载
                tensor = load_single_image(full_path)
                fixed = fit_to_canvas(tensor, shrink_multiple=缩小倍数)
                images.append(fixed)

                # 遮罩逻辑保留（batch_load 无遮罩，按需对齐）
                _, h, w, _ = fixed.shape
                mask = torch.ones((1, h, w), dtype=torch.float32)
                masks.append(mask)
            except Exception as e:
                print(f"[凤希] 加载失败：{full_path} => {e}")

        if not images:
            return (None, None, None, 0)

        image_batch = torch.cat(images, dim=0)
        mask_batch = torch.cat(masks, dim=0)

        # 网格拼接（和 batch_load 对齐：auto 或 fixed_2）
        merged = grid_concat_images(images, cols_mode="fixed_2")
        
        return (image_batch, mask_batch, merged, len(images))