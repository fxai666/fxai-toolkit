import os

# 支持的图片格式
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

class FxAiGetImagePath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片文件夹路径": ("STRING", {"multiline": False}),
                "图片索引": ("STRING", {"default": "0", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("图片路径字符串(逗号分隔)", "选中图片数量")
    
    FUNCTION = "load_image"
    CATEGORY = "凤希AI/工具"

    def load_image(self, 图片文件夹路径, 图片索引):
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
        
        # 无图片返回
        if total_images == 0:
            return ("", 0)

        # ===================== 索引解析逻辑 =====================
        index_str = 图片索引.strip()
        index_list = list(range(total_images))
        
        # 不等于-1则解析索引
        if index_str != "-1":
            index_list = []
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
                    idx = int(s)
                    index_list.append(idx)
        # ======================================================
        
        # 去重并保持顺序
        unique_indices = []
        seen = set()
        for idx in index_list:
            idx = idx % total_images
            if idx not in seen:
                unique_indices.append(idx)
                seen.add(idx)
        
        # 获取选中的图片路径
        selected_paths = []
        for idx in unique_indices:
            selected_paths.append(image_files[idx])
        
        # 转为 逗号分隔 字符串
        path_str = ",".join(selected_paths)
        count = len(selected_paths)
        
        return (path_str, count)