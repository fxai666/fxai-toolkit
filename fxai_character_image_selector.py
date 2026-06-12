import os
import torch
import folder_paths
from fxai_image_utils import load_single_image

def get_image_path(full_relative_path):
    comfy_root = folder_paths.base_path
    base_dir = "fxai/image"
    target_path = os.path.join(comfy_root, base_dir, full_relative_path)
    return target_path

class FxAiCharacterImageSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selected_files": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "layout": "hidden"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图片列表",)
    FUNCTION = "load_images"
    CATEGORY = "凤希AI/角色"

    def load_images(self, selected_files, unique_id=None):
        images = []
        if not selected_files.strip():
            raise RuntimeError("没有图片，请选择图片文件")

        path_list = [p.strip() for p in selected_files.split(",") if p.strip()]

        for rel_path in path_list:
            full_path = get_image_path(rel_path)
            if not os.path.exists(full_path):
                print(f"文件不存在，跳过: {full_path}")
                continue
            img_tensor = load_single_image(full_path)
            images.append(img_tensor)

        if not images:
            raise RuntimeError("没有图片，请选择图片文件")

        return (images,)