import cv2
import numpy as np
import torch
from PIL import Image


class FxAiImageLoopTile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "输出宽度": ("INT", {"default": 736, "min": 32, "max": 8192, "step": 32}),
                "输出高度": ("INT", {"default": 1280, "min": 32, "max": 8192, "step": 32}),
                "总帧数": ("INT", {"default": 17, "min": 1, "max": 1000, "step": 8}),
                "图片序列": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出序列",)
    FUNCTION = "process"
    CATEGORY = "凤希AI/工具"

    def process(self, 输出宽度, 输出高度, 总帧数, 图片序列):
        processed_images = []
        total = 图片序列.shape[0]
        
        for i in range(total):
            img = 图片序列[i]
            resized = self._prepare_image(img, (输出宽度, 输出高度))
            processed_images.append(resized)

        frames = self._expand_frames(processed_images, 总帧数)
        output = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
        return (output,)

    @staticmethod
    def _tensor_to_rgb(image):
        if isinstance(image, torch.Tensor):
            if image.ndim == 4:
                image = image[0]
            image = image.detach().cpu().numpy()

        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]

        return np.ascontiguousarray(image)

    @staticmethod
    def _prepare_image(image, target_size):
        img_array = FxAiImageLoopTile._tensor_to_rgb(image)
        pil_img = Image.fromarray(img_array).convert("RGB")
        img_array = np.array(pil_img)
        
        if img_array.shape[1] == target_size[0] and img_array.shape[0] == target_size[1]:
            return np.ascontiguousarray(img_array)
        
        return cv2.resize(img_array, target_size, interpolation=cv2.INTER_LANCZOS4)

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