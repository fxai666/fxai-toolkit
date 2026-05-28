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
        if not unique_id:
            return {"ui": {}, "result": ()}
        
        cache_name = f"cache_{unique_id}"
        img_path = os.path.join(CACHE_DIR, f"{cache_name}.png")
        ui = {}

        # 保存并输出图片
        images = []
        for batch in 图片:
            np_img = batch.cpu().numpy()
            np_img = np.clip(np_img, 0, 1)
            img = Image.fromarray((np_img * 255).astype(np.uint8))
            img.save(img_path)
            images.append({
                "filename": f"{cache_name}.png",
                "subfolder": "persist_preview",
                "type": "temp"
            })
        ui["images"] = images

        # 读取缓存，切换标签/刷新自动恢复
        if os.path.exists(img_path):
            ui["images"] = [{
                "filename": f"{cache_name}.png",
                "subfolder": "persist_preview",
                "type": "temp"
            }]

        return {"ui": ui, "result": ()}