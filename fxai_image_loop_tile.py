import torch
import numpy as np
from fxai_image_utils import ImageSizeController


class FxAiImageLoopTile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "输出宽度": ("INT", {"default": 704, "min": 32, "max": 8192, "step": 32}),
                "输出高度": ("INT", {"default": 1280, "min": 32, "max": 8192, "step": 8}),
                "总帧数": ("INT", {"default": 241, "min": 1, "max": 1800, "step": 8}),
                "图片序列": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE","INT",)
    RETURN_NAMES = ("输出序列","总帧数",)
    FUNCTION = "process"
    CATEGORY = "凤希AI/图片"

    def process(self, 输出宽度, 输出高度, 总帧数, 图片序列):
        img_ctrl = ImageSizeController(canvas_w=输出宽度, canvas_h=输出高度, bg_color=(255,255,255))

        if isinstance(图片序列, list):
            frame_list = []
            for item in 图片序列:
                frame_list.append(item if item.dim() == 4 else item.unsqueeze(0))
        else:
            frame_list = [图片序列[i:i+1] for i in range(图片序列.shape[0])]

        processed = [img_ctrl.crop_fill_to_canvas(f) for f in frame_list]
        processed = torch.cat(processed, dim=0)

        frames_np = (processed.cpu().numpy() * 255).astype(np.uint8)
        final_frames = self._expand_frames(frames_np, 总帧数)
        output = torch.from_numpy(np.stack(final_frames).astype(np.float32) / 255.0)

        return (output, 总帧数,)

    @staticmethod
    def _expand_frames(images, target_frames):
        cnt = len(images)
        base, rem = divmod(target_frames, cnt)
        out = []
        for i, img in enumerate(images):
            out += [img] * (base + (1 if i < rem else 0))
        return out