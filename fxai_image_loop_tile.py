import torch
import numpy as np
from fxai_image_utils import ImageSizeController  # 直接调用你的工具类


class FxAiImageLoopTile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "输出宽度": ("INT", {"default": 704, "min": 32, "max": 8192, "step": 32}),
                "输出高度": ("INT", {"default": 1280, "min": 32, "max": 8192, "step": 32}),
                "总帧数": ("INT", {"default": 241, "min": 1, "max": 1800, "step": 8}),
                "图片序列": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出序列",)
    FUNCTION = "process"
    CATEGORY = "凤希AI/图片"

    def process(self, 输出宽度, 输出高度, 总帧数, 图片序列):
        img_ctrl = ImageSizeController(canvas_w=输出宽度, canvas_h=输出高度,bg_color=(255,255,255))
        
        processed = []
        for i in range(图片序列.shape[0]):
            img = 图片序列[i:i+1]
            processed_img = img_ctrl.crop_fill_to_canvas(img)
            processed.append(processed_img)

        # 合并处理后的图片
        processed = torch.cat(processed, dim=0)
        
        # 转 numpy 用于循环平铺
        frames_np = (processed.cpu().numpy() * 255).astype(np.uint8)
        
        # 原版循环平铺逻辑（不变）
        final_frames = self._expand_frames(frames_np, 总帧数)
        
        # 转回 ComfyUI 标准张量
        output = torch.from_numpy(np.stack(final_frames).astype(np.float32) / 255.0)
        return (output,)

    @staticmethod
    def _expand_frames(images, target_frames):
        count = len(images)
        base = target_frames // count
        rem = target_frames % count
        
        frames = []
        for idx, img in enumerate(images):
            repeat = base + (1 if idx < rem else 0)
            frames.extend([img] * repeat)
        return frames


NODE_CLASS_MAPPINGS = {
    "FxAiImageLoopTile": FxAiImageLoopTile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FxAiImageLoopTile": "凤希AI 图片循环平铺",
}