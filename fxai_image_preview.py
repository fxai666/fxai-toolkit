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
        # 空值判断
        if 图片 is None or not unique_id:
            return {"ui": {}, "result": ()}

        ui = {}
        images = []

        # 遍历 batch 中每一张图片，使用不同文件名保存
        for i, batch in enumerate(图片):
            # 每张图片使用独立文件名，避免覆盖
            cache_name = f"cache_{unique_id}_{i}"
            img_filename = f"{cache_name}.png"
            img_path = os.path.join(CACHE_DIR, img_filename)

            # 张量转图片
            np_img = batch.cpu().numpy()
            np_img = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)
            img = Image.fromarray(np_img)
            img.save(img_path)

            # 加入 UI 显示列表
            images.append({
                "filename": img_filename,
                "subfolder": "persist_preview",
                "type": "temp"
            })

        ui["images"] = images
        return {"ui": ui, "result": ()}