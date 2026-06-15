import torch
from PIL import Image
import numpy as np
import base64
from io import BytesIO

class FxAiImageLoadBase64:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Base64": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("图片", "遮罩", "宽度", "高度")
    FUNCTION = "load_image"
    CATEGORY = "凤希AI/图片"

    def load_image(self, Base64):
        if "," in Base64:
            b64_content = Base64.split(",", 1)[1]
        else:
            b64_content = Base64

        img_bytes = base64.b64decode(b64_content)
        pil_img = Image.open(BytesIO(img_bytes))
        w, h = pil_img.size

        if pil_img.mode == "RGBA":
            r, g, b, a = pil_img.split()
            mask_np = np.array(a, dtype=np.float32) / 255.0
            pil_img = Image.merge("RGB", (r, g, b))
        else:
            mask_np = np.ones((h, w), dtype=np.float32)
        
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        img_np = np.array(pil_img, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(img_np).unsqueeze(0)

        return (image_tensor, mask_tensor, w, h)