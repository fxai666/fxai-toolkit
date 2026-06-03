import os
import torch
from fxai_image_utils import (
    load_single_image, fit_to_canvas, grid_concat_images, IMAGE_EXTENSIONS
)

class FxAiCharacterBatchLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片文件夹路径": ("STRING", {"multiline": False}),
                "图片索引": ("STRING", {"default": "0", "multiline": False}),
                "缩小倍数": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE","IMAGE","INT")
    RETURN_NAMES = ("图片列表", "网格拼接大图", "总数量")
    FUNCTION = "load_image"
    CATEGORY = "凤希AI/角色"

    def load_image(self, 图片文件夹路径, 图片索引, 缩小倍数):
        folder_path = 图片文件夹路径.strip()
        if not os.path.isdir(folder_path):
            raise RuntimeError(f"文件夹不存在：{folder_path}")
        
        # 筛选图片文件
        image_files = []
        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                full_path = os.path.join(folder_path, filename)
                image_files.append(full_path)
        
        total_images = len(image_files)
        if total_images == 0:
            return (None, None, 0)

        # 解析图片索引
        index_str = 图片索引.strip()
        index_list = list(range(total_images)) if index_str == "-1" else []
        if index_str != "-1":
            for s in index_str.split(','):
                s = s.strip()
                if not s:
                    continue
                if ':' in s:
                    try:
                        start, end = map(int, s.split(':'))
                        end = end + 1
                        start = max(0, start)
                        end = min(total_images, end)
                        index_list.extend(range(start, end))
                    except ValueError:
                        raise RuntimeError(f"索引范围格式错误：{s}")
                else:
                    if not s.isdigit():
                        raise RuntimeError(f"索引必须是数字或范围：{s}")
                    index_list.append(int(s))
        
        # 去重并处理负数索引
        unique_indices = []
        seen = set()
        for idx in index_list:
            idx = idx % total_images
            if idx not in seen:
                unique_indices.append(idx)
                seen.add(idx)
        
        # 加载图片（统一使用工具函数）
        images = []
        for idx in unique_indices:
            try:
                img_path = image_files[idx]
                img_tensor = load_single_image(img_path)
                fixed_img = fit_to_canvas(img_tensor, shrink_multiple=缩小倍数)
                images.append(fixed_img)
            except Exception as e:
                print(f"加载图片失败 {image_files[idx]}: {str(e)}")
                continue
        
        if not images:
            return (None, None, 0)

        merged_image = grid_concat_images(images, cols_mode="auto")
        merged_image = fit_to_canvas(merged_image, shrink_multiple=缩小倍数)
        
        return (images, merged_image, len(images))