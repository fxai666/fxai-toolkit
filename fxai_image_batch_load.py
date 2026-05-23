import os
import torch
from PIL import Image
import numpy as np
import math

# 工具函数：加载单张图片并按整数倍数缩小（完全没改！）
def load_single_image(image_path, shrink_multiple=1):
    img = Image.open(image_path).convert("RGB")
    
    # 整数倍数缩小
    if shrink_multiple > 1:
        new_width = int(img.width / shrink_multiple)
        new_height = int(img.height / shrink_multiple)
        # 防止缩小后尺寸为0导致报错
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np)[None,]  # [1, H, W, 3]
    return img_tensor

# 支持的图片格式（完全没改！）
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
    
    # 已删除 OUTPUT_IS_LIST （保持你原来的正确设置）
    FUNCTION = "load_image"
    CATEGORY = "凤希AI/图片"

    def load_image(self, 图片文件夹路径, 图片索引, 缩小倍数):
        folder_path = 图片文件夹路径.strip()
        
        # ===================== 这里 100% 是你原来的代码！完全没动！！ =====================
        if not os.path.isdir(folder_path):
            raise RuntimeError(f"文件夹不存在：{folder_path}")
        
        # 排序+过滤图片
        image_files = []
        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                full_path = os.path.join(folder_path, filename)
                image_files.append(full_path)
        
        total_images = len(image_files)
        if total_images == 0:
            raise RuntimeError("文件夹内没有图片")
        
        # 解析索引
        index_list = []
        for s in 图片索引.split(','):
            s = s.strip()
            if not s:
                continue  # 跳过空字符串
            
            if ':' in s:  # 支持范围格式如 "0:5"
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
        
        # 去除重复索引并保持顺序
        unique_indices = []
        seen = set()
        for idx in index_list:
            idx = idx % total_images  # 自动循环，防止越界
            if idx not in seen:
                unique_indices.append(idx)
                seen.add(idx)
        
        images = []
        masks = []
        
        for idx in unique_indices:
            if idx >= total_images:
                continue
                
            try:
                img_path = image_files[idx]
                img = load_single_image(img_path, 缩小倍数)
                _, h, w, _ = img.shape
                mask = torch.ones((1, h, w), dtype=torch.float32)
                
                images.append(img)
                masks.append(mask)
            except Exception as e:
                print(f"加载图片失败 {image_files[idx]}: {str(e)}")
                continue
        
        if not images:
            raise RuntimeError("没有成功加载任何图片")
        # ================================================================================

        # ===================== 只在这里新增：自动网格拼接（兼容奇数张） =====================
        def grid_concat(images_list):
            n = len(images_list)
            if n == 1:
                return images_list[0]

            # 自动排版：接近正方形
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)

            # 统一尺寸
            target_h = images_list[0].shape[1]
            target_w = images_list[0].shape[2]
            resized = []
            for img in images_list:
                pil = Image.fromarray((img.squeeze(0).cpu().numpy() * 255).astype(np.uint8))
                pil = pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
                t = torch.from_numpy(np.array(pil).astype(np.float32) / 255.0)[None,]
                resized.append(t)

            # 空白填充（解决3张、5张、7张奇数问题）
            blank = torch.zeros_like(resized[0])
            while len(resized) < rows * cols:
                resized.append(blank)

            # 拼接
            rows_tensor = []
            for i in range(rows):
                row = resized[i*cols : (i+1)*cols]
                rows_tensor.append(torch.cat(row, dim=2))
            return torch.cat(rows_tensor, dim=1)
        
        merged_image = grid_concat(images)
        # ====================================================================================

        # 返回格式完全保持你原来的正确写法！
        return ((images,), (masks,), merged_image, len(images))