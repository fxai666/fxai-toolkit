import os
import torch
from PIL import Image
import numpy as np
import math

# 固定统一画布尺寸
CANVAS_W = 1024
CANVAS_H = 1024

# 工具函数：加载单张图片并按整数倍数缩小
def load_single_image(image_path, shrink_multiple=1):
    img = Image.open(image_path).convert("RGB")
    
    # 整数倍数缩小
    if shrink_multiple > 1:
        new_width = int(img.width / shrink_multiple)
        new_height = int(img.height / shrink_multiple)
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np)[None,]  # [1, H, W, 3]
    return img_tensor

# 等比例缩放到1024*1024画布，居中不拉伸
def fit_to_canvas(tensor_img):
    img = tensor_img.squeeze(0).cpu().numpy()
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    src_w, src_h = pil_img.size

    # 等比例缩放
    scale = min(CANVAS_W / src_w, CANVAS_H / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 黑色画布居中粘贴
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (0, 0, 0))
    offset_x = (CANVAS_W - new_w) // 2
    offset_y = (CANVAS_H - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    out_np = np.array(canvas).astype(np.float32) / 255.0
    return torch.from_numpy(out_np)[None,]

# 支持的图片格式
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

class FxAiImageBatchLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片文件夹路径": ("STRING", {"multiline": False}),
                "图片索引": ("STRING", {"default": "0", "multiline": False}),
                "缩小倍数": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK","IMAGE","INT")
    RETURN_NAMES = ("图片列表", "遮罩列表", "网格拼接大图", "总数量")
    
    FUNCTION = "load_image"
    CATEGORY = "凤希AI/图片"

    def load_image(self, 图片文件夹路径, 图片索引, 缩小倍数):
        folder_path = 图片文件夹路径.strip()
        
        if not os.path.isdir(folder_path):
            raise RuntimeError(f"文件夹不存在：{folder_path}")
        
        # 排序+过滤图片
        image_files = []
        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                full_path = os.path.join(folder_path, filename)
                image_files.append(full_path)
        
        total_images = len(image_files)
        
        # ========== 没图片，直接返回 None，干净利落 ==========
        if total_images == 0:
            return (None, None, None, 0)

        CANVAS_W = 1024 // 缩小倍数
        CANVAS_H = 1024 // 缩小倍数

        # ===================== 核心优化逻辑 =====================
        index_str = 图片索引.strip()
        # 初始化：默认加载所有图片
        index_list = list(range(total_images))
        
        # 如果不等于-1，才执行原来的索引解析逻辑
        if index_str != "-1":
            index_list = []  # 清空，重新解析
            for s in index_str.split(','):
                s = s.strip()
                if not s:
                    continue
                
                if ':' in s:
                    try:
                        start, end = map(int, s.split(':'))
                        end = end + 1
                        if start < 0:
                            start = 0
                        if end > total_images:
                            end = total_images
                        index_list.extend(range(start, end))
                    except ValueError:
                        raise RuntimeError(f"索引范围格式错误：{s}")
                else:
                    if not s.isdigit():
                        raise RuntimeError(f"索引必须是数字或范围：{s}")
                    idx = int(s)
                    index_list.append(idx)
        # ======================================================
        
        # 去除重复索引并保持顺序
        unique_indices = []
        seen = set()
        for idx in index_list:
            idx = idx % total_images
            if idx not in seen:
                unique_indices.append(idx)
                seen.add(idx)
        
        images = []
        masks = []
        
        for idx in unique_indices:
            try:
                img_path = image_files[idx]
                img = load_single_image(img_path, 缩小倍数)
                fixed_img = fit_to_canvas(img)
                _, h, w, _ = fixed_img.shape
                mask = torch.ones((1, h, w), dtype=torch.float32)
                
                images.append(fixed_img)
                masks.append(mask)
            except Exception as e:
                print(f"加载图片失败 {image_files[idx]}: {str(e)}")
                continue
        
        if not images:
            return (None, None, None, 0)

        # 自动网格拼接
        def grid_concat(images_list):
            n = len(images_list)
            if n == 1:
                return images_list[0]

            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)

            blank = torch.zeros_like(images_list[0])
            while len(images_list) < rows * cols:
                images_list.append(blank)

            rows_tensor = []
            for i in range(rows):
                row = images_list[i*cols : (i+1)*cols]
                rows_tensor.append(torch.cat(row, dim=2))
            return torch.cat(rows_tensor, dim=1)
        
        merged_image = grid_concat(images)
        return ((images,), (masks,), merged_image, len(images))