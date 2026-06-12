import os
import torch
import numpy as np
from PIL import Image
import folder_paths

CACHE_DIR = os.path.join(folder_paths.temp_directory, "persist_preview")
os.makedirs(CACHE_DIR, exist_ok=True)

class FxAiImagePreview:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "凤希AI/图片"

    def preview(self, 图片, unique_id=None):
        if 图片 is None or not unique_id:
            return {"ui": {}, "result": ()}

        ui = {}
        images = []
        img_index = 0

        tensor_list = []
        if isinstance(图片, list):
            for t in 图片:
                if isinstance(t, torch.Tensor) and len(t.shape) == 4:
                    tensor_list.append(t)
        elif isinstance(图片, torch.Tensor) and len(图片.shape) == 4:
            tensor_list.append(图片)

        # 遍历所有批次张量，逐图保存
        for batch_tensor in tensor_list:
            for _, single_img_tensor in enumerate(batch_tensor):
                cache_name = f"cache_{unique_id}_{img_index}"
                img_filename = f"{cache_name}.png"
                img_path = os.path.join(CACHE_DIR, img_filename)

                np_img = single_img_tensor.cpu().numpy()
                np_img = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)
                img = Image.fromarray(np_img)
                img.save(img_path)

                images.append({
                    "filename": img_filename,
                    "subfolder": "persist_preview",
                    "type": "temp"
                })
                img_index += 1

        ui["images"] = images
        return {"ui": ui, "result": ()}