import os
import torch
from PIL import Image
import numpy as np

# 工具函数：加载单张图片（ComfyUI 标准格式）
def load_single_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np)[None,]  # 转为 [1, H, W, 3]
    return img_tensor

# 支持的图片格式
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

class FxAiLoadImageByIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片文件夹路径": ("STRING", {"multiline": False}),
                "图片索引": ("INT", {"default": 0, "min": 0}),
            },
            "optional": {
                "刷新标记": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT")
    RETURN_NAMES = ("图片", "遮罩", "当前图片路径", "总图片数量")
    FUNCTION = "load_image"
    CATEGORY = "凤希AI"

    def load_image(self, 图片文件夹路径, 图片索引, 刷新标记=0):
        # 1. 清理路径
        folder_path = 图片文件夹路径.strip()
        
        # 2. 检查文件夹是否存在
        if not os.path.isdir(folder_path):
            raise RuntimeError(f"文件夹不存在：{folder_path}")
        
        # 3. 获取文件夹里所有图片（按文件名排序）
        image_files = []
        for f in os.listdir(folder_path):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                full_path = os.path.join(folder_path, f)
                image_files.append(full_path)
        
        # 按文件名排序（保证每次顺序一致）
        image_files.sort()
        
        # 4. 检查是否有图片
        total_images = len(image_files)
        if total_images == 0:
            raise RuntimeError(f"文件夹中没有找到图片：{folder_path}")
        
        # 5. 检查索引是否越界
        if 图片索引 >= total_images:
            raise RuntimeError(f"索引越界！共 {total_images} 张图，索引范围：0 ~ {total_images-1}")
        
        # 6. 加载选中的图片
        target_path = image_files[图片索引]
        image_tensor = load_single_image(target_path)
        
        # 7. 生成遮罩
        h, w = image_tensor.shape[1], image_tensor.shape[2]
        mask_tensor = torch.ones((1, h, w), dtype=torch.float32)
        
        return (image_tensor, mask_tensor, target_path, total_images)