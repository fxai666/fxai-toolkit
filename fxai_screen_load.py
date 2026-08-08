import os
import re
import math
import folder_paths
from fxai_image_utils import load_single_image

def get_image_dir(subdir=""):
    comfy_root = folder_paths.base_path
    base_dir = "fxai/image"
    target_dir = os.path.join(comfy_root, base_dir)

    if subdir:
        subdir = re.sub(r'[\\/*?:"<>|]', "", subdir)
        target_dir = os.path.join(target_dir, subdir)

    return target_dir

class FxAiScreenLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "场景数据": ("LIST", {"forceInput": True}),
                "行索引": ("INT", {"forceInput": True}),
            },
            "optional": {
                "通用提示词": ("STRING", {"default": "", "forceInput": True}),
                "尾部通用提示词": ("STRING", {"default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "INT")
    RETURN_NAMES = ("台词", "素材", "帧数")

    FUNCTION = "get_scene_data"
    CATEGORY = "凤希AI/影视剧场"

    def get_scene_data(self, 场景数据, 行索引, 通用提示词="", 尾部通用提示词=""):
        台词 = ""
        提示词 = ""
        时长 = 15.0
        frame = 0
        images = []

        try:
            total_lines = len(场景数据) if isinstance(场景数据, list) else 0

            if 行索引 < 0 or total_lines == 0 or 行索引 >= total_lines:
                raise IndexError(f"行索引越界：{行索引} / 总行数：{total_lines}")

            item = 场景数据[行索引]
            台词 = item.get("台词", item.get("提示词文本", ""))
            提示词 = 通用提示词 + 台词 + 尾部通用提示词
            素材 = item.get("素材", "")
            时长 = float(item.get("时长", 15.0))

            path_list = [p.strip() for p in str(素材).split(",") if p.strip()]
            x = max(5, round(时长 * 24))
            frame = x + (5 - x) % 17

        except Exception as e:
            print(f"✅ [凤希AI场景加载] 异常：{e}")
            x = max(5, int(round(时长 * 24)))
            frame = x + (5 - x) % 17
            return (
                f"{通用提示词}{台词 or ''}{尾部通用提示词}",
                images,
                frame,
            )

        for rel_path in path_list:
            parts = rel_path.split("/", 1)
            if len(parts) != 2:
                continue

            subdir, filename = parts
            full_dir = get_image_dir(subdir)
            full_path = os.path.join(full_dir, filename)

            if not os.path.exists(full_path):
                print(f"[凤希] 图片不存在：{full_path}")
                continue

            try:
                tensor = load_single_image(full_path)
                images.append(tensor)
            except Exception as e:
                print(f"[凤希] 加载失败：{full_path} => {e}")

        return (
            提示词,
            images,
            frame,
        )