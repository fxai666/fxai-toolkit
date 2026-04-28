import torch
import torch.nn.functional as F

class FxAiImageSizeConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "图片": ("IMAGE",),
                "最大边长": ("INT", {"default": 960, "min": 64, "max": 8192}),
                "整除系数": ("INT", {"default": 8, "min": 1, "max": 256}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("处理后图片", "实际宽度", "实际高度")
    FUNCTION = "process"
    CATEGORY = "凤希AI/工具"

    def process(self, 图片, 最大边长, 整除系数):
        batch, orig_h, orig_w, c = 图片.shape

        # 等比例缩放最长边到最大边长
        scale = 最大边长 / max(orig_w, orig_h)
        w_scaled = int(orig_w * scale)
        h_scaled = int(orig_h * scale)

        # 宽高调整为能被整除系数整除（不裁剪，只调整尺寸）
        w_scaled = ((w_scaled + 整除系数 - 1) // 整除系数) * 整除系数
        h_scaled = ((h_scaled + 整除系数 - 1) // 整除系数) * 整除系数
        
        # 保证最小尺寸
        w_scaled = max(整除系数, w_scaled)
        h_scaled = max(整除系数, h_scaled)

        # 缩放图片
        img = 图片.permute(0, 3, 1, 2)
        img = F.interpolate(img, size=(h_scaled, w_scaled), mode="bilinear", align_corners=False)
        img = img.permute(0, 2, 3, 1)

        # ✅ 移除了正方形裁剪，保持原图比例
        return (img, w_scaled, h_scaled)