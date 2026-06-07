# fxai_image_utils.py 重构版
import os
import math
import torch
import numpy as np
from PIL import Image

class ImageSizeController:
    """图片宽高控制类 - 统一管理图片尺寸、缩放、拼接逻辑"""
    # 默认画布尺寸（可通过初始化覆盖）
    DEFAULT_CANVAS_W = 1024
    DEFAULT_CANVAS_H = 1024
    # 支持的图片格式
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

    def __init__(self, canvas_w=None, canvas_h=None):
        """
        初始化图片尺寸控制器
        :param canvas_w: 自定义画布宽度（默认1024）
        :param canvas_h: 自定义画布高度（默认1024）
        """
        self.canvas_w = canvas_w or self.DEFAULT_CANVAS_W
        self.canvas_h = canvas_h or self.DEFAULT_CANVAS_H

    def load_single_image(self, image_path):
        """加载单张图片为张量（仅加载，不缩放）"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片路径不存在: {image_path}")
        if not image_path.lower().endswith(self.IMAGE_EXTENSIONS):
            raise ValueError(f"不支持的图片格式，仅支持: {self.IMAGE_EXTENSIONS}")
        
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np)[None,]  # [1, H, W, 3]
        return img_tensor

    def fit_to_canvas(self, tensor_img, shrink_multiple=1):
        """
        等比例缩放到指定画布（缩小倍数驱动），居中不拉伸
        :param tensor_img: 输入图片张量 [1, H, W, 3]
        :param shrink_multiple: 画布缩小倍数（如2则画布为 canvas_w/2, canvas_h/2）
        :return: 缩放后张量 [1, target_h, target_w, 3]
        """
        # 动态计算目标画布尺寸（基于实例化的宽高）
        target_w = self.canvas_w // shrink_multiple
        target_h = self.canvas_h // shrink_multiple
        
        # 获取原图尺寸
        _, src_h, src_w, _ = tensor_img.shape
        
        # 核心优化：判断尺寸是否已匹配，匹配则直接返回原张量
        if src_w == target_w and src_h == target_h:
            return tensor_img

        # 张量转PIL图片
        img = tensor_img.squeeze(0).cpu().numpy()
        pil_img = Image.fromarray((img * 255).astype(np.uint8))

        # 等比例缩放（避免拉伸）
        scale = min(target_w / src_w, target_h / src_h)
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)
        resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 黑色画布居中粘贴
        canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y))

        # 转回张量
        out_np = np.array(canvas).astype(np.float32) / 255.0
        return torch.from_numpy(out_np)[None,]

    def grid_concat_images(self, images_list, cols_mode="auto"):
        """
        网格拼接图片列表（基于当前画布尺寸适配）
        :param images_list: 图片张量列表（每个元素是 [1, H, W, 3]）
        :param cols_mode: "auto"（自适应列数）或 "fixed_2"（固定2列）
        :return: 拼接后的大图张量 [1, total_h, total_w, 3]
        """
        n = len(images_list)
        if n == 0:
            raise ValueError("图片列表不能为空")
        if n == 1:
            return images_list[0]

        # 列数逻辑
        if cols_mode == "fixed_2":
            cols = 2
        else:  # auto：基于画布尺寸自适应（接近正方形）
            cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        # 补空白图（匹配当前画布单张尺寸）
        blank = torch.zeros_like(images_list[0])
        imgs_copy = images_list.copy()
        while len(imgs_copy) < rows * cols:
            imgs_copy.append(blank)

        # 按行列拼接
        rows_tensor = []
        for i in range(rows):
            row_imgs = imgs_copy[i*cols : (i+1)*cols]
            rows_tensor.append(torch.cat(row_imgs, dim=2))  # 横向拼接（宽叠加）
        return torch.cat(rows_tensor, dim=1)  # 纵向拼接（高叠加）

# 保留原有全局函数，适配老代码
_global_size_controller = ImageSizeController()
load_single_image = _global_size_controller.load_single_image
fit_to_canvas = _global_size_controller.fit_to_canvas
grid_concat_images = _global_size_controller.grid_concat_images
IMAGE_EXTENSIONS = ImageSizeController.IMAGE_EXTENSIONS