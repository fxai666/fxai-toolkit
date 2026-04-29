import torch
from PIL import Image
import numpy as np

# 正确 4 步流程：先缩放 → 再对齐 → 再裁剪缩放图
def resize_image_final(image, target_max, base=32):
    # ComfyUI 图片格式：[batch, height, width, channel]
    h, w = image.shape[1:3]
    
    # 1. 等比缩放到最大边长（必须先做这一步）
    scale = target_max / max(w, h)
    sw, sh = int(round(w * scale)), int(round(h * scale))  # 四舍五入更精准
    
    # 先执行等比缩放（核心修复点）
    # permute 转换为 [batch, channel, height, width] 以支持 torch.nn.functional.interpolate
    img_tensor = image.permute(0, 3, 1, 2)
    scaled_img = torch.nn.functional.interpolate(
        img_tensor, size=(sh, sw), mode="bilinear", align_corners=False
    )
    # 转回 ComfyUI 格式 [batch, height, width, channel]
    scaled_img = scaled_img.permute(0, 2, 3, 1)
    
    # 2. 对【缩放后的尺寸】向下对齐基数
    fw = (sw // base) * base
    fh = (sh // base) * base

    # 3. 计算多余部分（基于缩放图的宽高）
    dw = sw - fw
    dh = sh - fh
    
    # 4. 居中裁剪【缩放后的图片】（不是原图！）
    left = dw // 2
    top = dh // 2
    cropped = scaled_img[:, top:top+fh, left:left+fw, :]
    
    return cropped, fw, fh

# ComfyUI 最终节点
class FxAiImageSizeConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片": ("IMAGE",),
                "目标最大边长": ("INT", {"default": 980, "min": 64, "max": 4096, "step": 1}),
                "对齐基数": ("INT", {"default": 32, "min": 2, "max": 128, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("输出图片", "宽度", "高度")
    FUNCTION = "process"
    CATEGORY = "凤希AI/图片"

    def process(self, 图片, 目标最大边长, 对齐基数):
        try:
            if 图片 is None or 图片.numel() == 0:
                return (None, 0, 0)
            
            out_img, w, h = resize_image_final(图片, 目标最大边长, 对齐基数)
            return (out_img, w, h)
        except Exception as e:
            raise RuntimeError(f"处理失败：{str(e)}")

# 节点注册
NODE_CLASS_MAPPINGS = {
    "FxAiImageSizeConfig": FxAiImageSizeConfig
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FxAiImageSizeConfig": "凤希AI 图片尺寸等比配置"
}