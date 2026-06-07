import os
import torch
import folder_paths
from fxai_image_utils import load_single_image, ImageSizeController  # 新增导入

def get_image_path(full_relative_path):
    comfy_root = folder_paths.base_path
    base_dir = "fxai/image"
    target_path = os.path.join(comfy_root, base_dir, full_relative_path)
    return target_path

class FxAiCharacterImageSelector:
    def __init__(self):
        self.images = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selected_files": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "layout": "hidden"
                }),
                "宽度": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "高度": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图片序列",)
    FUNCTION = "load_images"
    CATEGORY = "凤希AI/角色"

    def load_images(self, selected_files, 宽度, 高度, unique_id=None):
        images = []
        print(f"{selected_files}");
        if not selected_files.strip():
            return (torch.zeros((1, 64, 64, 3)), )

        size_controller = ImageSizeController(canvas_w=宽度, canvas_h=高度)
        
        path_list = [p.strip() for p in selected_files.split(",") if p.strip()]

        for rel_path in path_list:
            full_path = get_image_path(rel_path)

            if not os.path.exists(full_path):
                continue

            img = size_controller.load_single_image(full_path)
            img = size_controller.fit_to_canvas(img)
            images.append(img)

        if images:
            self.images = torch.cat(images, dim=0)
        else:
            self.images = torch.zeros((1, 64, 64, 3))

        return (self.images,)