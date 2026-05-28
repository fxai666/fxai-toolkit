import os
import torch
import numpy as np
from PIL import Image
import folder_paths

CACHE_DIR = os.path.join(folder_paths.temp_directory, "persist_preview")
os.makedirs(CACHE_DIR, exist_ok=True)

class FxAiAnyPreview:
    def __init__(self):
        self.img_cache_file = None
        self.text_cache = None
        self.text_widget = None  # 用于显示文本的widget

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "任意输入": ("*",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "凤希AI/工具"

    def preview(self, 任意输入, unique_id=None):
        if not unique_id:
            return {"ui": {}, "result": ()}
        
        cache_name = f"cache_{unique_id}"
        img_path = os.path.join(CACHE_DIR, f"{cache_name}.png")
        txt_path = os.path.join(CACHE_DIR, f"{cache_name}.txt")
        ui = {}

        # --------------------------
        # 1. 图片：走官方ui["images"]（正常显示）
        # --------------------------
        if isinstance(任意输入, torch.Tensor) and 任意输入.dim() == 4:
            images = []
            for batch in 任意输入:
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
            # 清空文本widget和缓存
            self.text_cache = None
            if os.path.exists(txt_path):
                os.remove(txt_path)
            return {"ui": ui, "result": ()}

        # --------------------------
        # 2. 文本类：写入缓存 + 存到实例变量（用于前端widget渲染）
        # --------------------------
        if isinstance(任意输入, (str, int, float, bool, list, dict)):
            text_val = str(任意输入)
            # 写入文件缓存（持久化）
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text_val)
            # 存在实例变量，前端会读这个来显示
            self.text_cache = text_val
            # 清空图片缓存
            if os.path.exists(img_path):
                os.remove(img_path)
            # 不返回ui["text"]/strings，避免冲突
            return {"ui": {}, "result": ()}

        # --------------------------
        # 3. 无新输入：恢复缓存
        # --------------------------
        if os.path.exists(img_path):
            ui["images"] = [{
                "filename": f"{cache_name}.png",
                "subfolder": "persist_preview",
                "type": "temp"
            }]
            self.text_cache = None
        elif os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                self.text_cache = f.read()

        return {"ui": ui, "result": ()}

    # --------------------------
    # 关键：前端会调用这个方法，把文本画在节点上
    # --------------------------
    def get_widgets(self):
        widgets = []
        if self.text_cache is not None:
            # 创建一个只读文本框，显示内容
            widgets.append({
                "name": "preview_text",
                "type": "text",
                "value": self.text_cache,
                "readonly": True,
                "multiline": True
            })
        return widgets