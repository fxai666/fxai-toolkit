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
    CATEGORY = "凤希AI/工具"

    def preview(self, 图片, unique_id=None):
        # 合并所有空值判断：图片为空 或 没有ID → 直接统一返回
        if 图片 is None or not unique_id:
            return {"ui": {}, "result": ()}

        # 下面只有正常情况才会执行
        cache_name = f"cache_{unique_id}"
        img_path = os.path.join(CACHE_DIR, f"{cache_name}.png")
        ui = {}
        images = []

        for batch in 图片:
            np_img = batch.cpu().numpy()
            np_img = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)
            img = Image.fromarray(np_img)
            img.save(img_path)
            images.append({
                "filename": f"{cache_name}.png",
                "subfolder": "persist_preview",
                "type": "temp"
            })

        if os.path.exists(img_path):
            ui["images"] = images

        return {"ui": ui, "result": ()}