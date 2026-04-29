import torch
from PIL import Image
import numpy as np

# ==============================
# 最高质量无损缩放（LANCZOS）
# ==============================
def resize_image_final(image, target_max, base=32):
    # ComfyUI 格式：[batch, height, width, channel]
    h, w = image.shape[1:3]
    
    # 1. 计算等比缩放后的尺寸
    scale = target_max / max(w, h)
    sw, sh = int(round(w * scale)), int(round(h * scale))

    # ==============================
    # 🔴 核心替换：最高质量无损缩放
    # ==============================
    # 张量 → PIL
    img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    
    # 最高画质缩放（无损、清晰）
    img = img.resize((sw, sh), Image.Resampling.LANCZOS)
    
    # PIL → 张量
    scaled_img_np = np.array(img).astype(np.float32) / 255.0
    scaled_img = torch.from_numpy(scaled_img_np).unsqueeze(0).to(image.device)

    # 2. 对齐基数（你原来逻辑不变）
    fw = (sw // base) * base
    fh = (sh // base) * base

    # 3. 居中裁剪（你原来逻辑不变）
    dw = sw - fw
    dh = sh - fh
    left = dw // 2
    top = dh // 2
    cropped = scaled_img[:, top:top+fh, left:left+fw, :]
    
    return cropped, fw, fh

# ==============================
# 你原来的节点完全不动
# ==============================
class FxAiImageSizeConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片": ("IMAGE",),
                "最大边长": ("INT", {"default": 980, "min": 64, "max": 4096, "step": 1}),
                "对齐基数": ("INT", {"default": 32, "min": 2, "max": 128, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("输出图片", "宽度", "高度")
    FUNCTION = "process"
    CATEGORY = "凤希AI/图片"

    def process(self, 图片, 最大边长, 对齐基数):
        try:
            if 图片 is None or 图片.numel() == 0:
                return (None, 0, 0)
            
            out_img, w, h = resize_image_final(图片, 最大边长, 对齐基数)
            return (out_img, w, h)
        except Exception as e:
            raise RuntimeError(f"处理失败：{str(e)}")