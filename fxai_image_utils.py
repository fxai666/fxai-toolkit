# fxai_image_utils.py
import os
import math
import torch
import numpy as np
from PIL import Image

# 全局统一画布尺寸（仅定义一次）
CANVAS_W = 1024
CANVAS_H = 1024

# 支持的图片格式（全局复用）
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

def load_single_image(image_path):
    """加载单张图片为张量（仅加载，不缩放）"""
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np)[None,]  # [1, H, W, 3]
    return img_tensor

def fit_to_canvas(tensor_img, shrink_multiple=1):
    """等比例缩放到指定画布（缩小倍数驱动），居中不拉伸"""
    # 计算缩放后的画布尺寸
    target_w = CANVAS_W // shrink_multiple
    target_h = CANVAS_H // shrink_multiple

    # 张量转PIL图片
    img = tensor_img.squeeze(0).cpu().numpy()
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    src_w, src_h = pil_img.size

    # 等比例缩放
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

def grid_concat_images(images_list, cols_mode="auto"):
    """
    网格拼接图片列表
    :param images_list: 图片张量列表（每个元素是 [1, H, W, 3]）
    :param cols_mode: "auto"（自适应列数）或 "fixed_2"（固定2列）
    :return: 拼接后的大图张量
    """
    n = len(images_list)
    if n == 1:
        return images_list[0]

    # 列数逻辑统一
    if cols_mode == "fixed_2":
        cols = 2
    else:  # auto
        cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # 补空白图
    blank = torch.zeros_like(images_list[0])
    imgs_copy = images_list.copy()
    while len(imgs_copy) < rows * cols:
        imgs_copy.append(blank)

    # 拼接行+列
    rows_tensor = []
    for i in range(rows):
        row = imgs_copy[i*cols : (i+1)*cols]
        rows_tensor.append(torch.cat(row, dim=2))
    return torch.cat(rows_tensor, dim=1)