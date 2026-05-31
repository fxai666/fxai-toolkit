import os
import re
import torch
import math
import folder_paths
from PIL import Image
import numpy as np

# 固定统一画布尺寸
CANVAS_W = 1024
CANVAS_H = 1024

def get_image_dir(subdir=""):
    comfy_root = folder_paths.base_path
    base_dir = "fxai/image"
    target_dir = os.path.join(comfy_root, base_dir)
    
    if subdir:
        subdir = re.sub(r'[\\/*?:"<>|]', "", subdir)
        target_dir = os.path.join(target_dir, subdir)
    
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

# 工具函数：加载单张图片
def load_single_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np)[None,]
    return img_tensor

# 等比例缩放到1024*1024画布，居中不拉伸
def fit_to_canvas(tensor_img, shrink_multiple=1):
    global CANVAS_W, CANVAS_H
    w = CANVAS_W // shrink_multiple
    h = CANVAS_H // shrink_multiple

    img = tensor_img.squeeze(0).cpu().numpy()
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    src_w, src_h = pil_img.size

    scale = min(w / src_w, h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    offset_x = (w - new_w) // 2
    offset_y = (h - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    out_np = np.array(canvas).astype(np.float32) / 255.0
    return torch.from_numpy(out_np)[None,]

# ==========================================
# 🔥 对接 JS 弹窗返回的路径
# ==========================================
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
            # rel_path 格式：tops/abc.png
            parts = rel_path.split("/", 1)
            if len(parts) != 2:
                continue

            subdir, filename = parts
            # 🔥 直接用你的路径函数
            full_dir = get_image_dir(subdir)
            full_path = os.path.join(full_dir, filename)

            if not os.path.exists(full_path):
                print(f"[凤希] 图片不存在：{full_path}")
                continue

            try:
                tensor = load_single_image(full_path)
                fixed = fit_to_canvas(tensor, 缩小倍数)
                images.append(fixed)

                # 遮罩
                _, h, w, _ = fixed.shape
                mask = torch.ones((1, h, w), dtype=torch.float32)
                masks.append(mask)
            except Exception as e:
                print(f"[凤希] 加载失败：{full_path} => {e}")

        if not images:
            return (None, None, None, 0)

        # 网格拼接
        def make_grid(imgs):
            n = len(imgs)
            if n == 1:
                return imgs[0]
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)
            blank = torch.zeros_like(imgs[0])
            while len(imgs) < rows * cols:
                imgs.append(blank)
            rows_list = [torch.cat(imgs[i*cols:(i+1)*cols], dim=2) for i in range(rows)]
            return torch.cat(rows_list, dim=1)

        merged = make_grid(images.copy())
        return ((images,), (masks,), merged, len(images))