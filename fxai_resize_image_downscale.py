import os
import torch
from PIL import Image
import numpy as np

# 工具函数：等比缩小图片（核心逻辑，适配1~100倍缩小 + 宽高必须能被2整除）
def resize_image_downscale(image_tensor, downscale_times):
    """
    等比缩小图片，缩小倍数为1~100（表示原始尺寸/缩小倍数）
    最终输出宽高强制能被 2 整除
    :param image_tensor: ComfyUI格式图片张量 [1, H, W, 3]
    :param downscale_times: 缩小倍数（1 ≤ downscale_times ≤ 100，整数）
    :return: 缩小后的图片张量、新高度、新宽度
    """
    # 校验缩小倍数合法性
    if not isinstance(downscale_times, int) or downscale_times < 1 or downscale_times > 100:
        raise RuntimeError(f"缩小倍数必须是1~100之间的整数！当前值：{downscale_times}")
    
    # 提取原始尺寸 (H, W, C)
    _, h, w, c = image_tensor.shape
    
    # 1. 计算等比缩小后的原始尺寸
    new_w = int(w / downscale_times)
    new_h = int(h / downscale_times)
    
    # 2. 核心：确保宽和高都能被 2 整除（向上取整）
    new_w = new_w + 1 if new_w % 2 != 0 else new_w
    new_h = new_h + 1 if new_h % 2 != 0 else new_h

    # 最小不能小于2
    new_w = max(new_w, 2)
    new_h = max(new_h, 2)
    
    # 将张量转回PIL Image进行等比缩放
    img_np = image_tensor[0].cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np).convert("RGB")
    
    # 缩放
    img_pil_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 转回张量
    img_resized_np = np.array(img_pil_resized).astype(np.float32) / 255.0
    img_resized_tensor = torch.from_numpy(img_resized_np)[None,]
    
    return img_resized_tensor, new_h, new_w

# 图片等比缩小节点（康复UI，1~100倍 + 宽高能被2整除）
class FxAiImageDownscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "图片": ("IMAGE",),
                "缩小倍数": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("缩小后图片", "缩小后高度", "缩小后宽度")
    FUNCTION = "downscale_image"
    CATEGORY = "凤希AI"

    def downscale_image(self, 图片=None, 缩小倍数=None):
        try:
            # ===================== 终极修复：空输入直接返回 None =====================
            if 图片 is None or 图片.numel() == 0:
                # 直接返回 None！下游节点会自动识别为空，不执行任何处理
                return (None, 0, 0)
            # ======================================================================
            
            resized_image, new_h, new_w = resize_image_downscale(图片, 缩小倍数)
            return (resized_image, new_h, new_w)
        except Exception as e:
            raise RuntimeError(f"图片缩小失败：{str(e)}")