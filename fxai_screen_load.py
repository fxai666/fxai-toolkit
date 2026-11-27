import os
import re
import math
import datetime
import folder_paths
from fxai_image_utils import load_single_image
from fxai_audio_utils import load_audio_tensor_from_file, slice_audio
from fxai_character_profile_manager import get_characters_by_avatars
from fxai_task_store import broadcast

_TRUE_WORDS = {"1", "true", "yes", "y", "on", "开", "开启", "是", "要", "续接"}


def _to_bool(v):
    """把场景行里的续接标记转成 bool。作者可能填 1/0、"true"、勾选框、中文"是"。"""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in _TRUE_WORDS
    return bool(v)


def scene_relay_flag(item):
    """读当前场景行的「场景续接」标记。

    场景续接按场景单独控制：每行自己决定本段要不要接上一段的续帧，
    所以这个开关挂在「影视剧场数据解析 / 场景加载器」上，
    由当前行数据决定，不再由生成控制器给一个全局总开关。
    """
    if not isinstance(item, dict):
        return False
    for key in ("场景续接", "启用场景续接", "续接", "scene_relay", "relay"):
        if key in item:
            return _to_bool(item[key])
    return False


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
                "音色剪切时长": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1,
                    "tooltip": "参考音色剪切秒数；0=按该行「时长」字段剪切，不足按音频实际长度"}),
            }
        }

    # 输出槽位只往后追加：第 6 槽放在最后，
    # 老工作流前 5 个槽位的连线一根都不用重连。
    RETURN_TYPES = ("STRING", "IMAGE", "INT", "LIST", "LIST", "BOOLEAN")
    RETURN_NAMES = ("台词", "素材", "帧数", "参考音色", "音频", "场景续接")

    FUNCTION = "get_scene_data"
    CATEGORY = "凤希AI/影视剧场"

    def get_scene_data(self, 场景数据, 行索引, 通用提示词="", 尾部通用提示词="", 音色剪切时长=0.0):
        台词 = ""
        提示词 = ""
        时长 = 15.0
        frame = 0
        images = []
        参考音色 = []
        音频 = []
        启用场景续接 = False

        try:
            total_lines = len(场景数据) if isinstance(场景数据, list) else 0

            if 行索引 < 0 or total_lines == 0 or 行索引 >= total_lines:
                raise IndexError(f"行索引越界：{行索引} / 总行数：{total_lines}")

            item = 场景数据[行索引]
            台词 = item.get("台词", item.get("提示词文本", ""))
            提示词 = 通用提示词 + 台词 + 尾部通用提示词
            素材 = item.get("素材", "")
            时长 = float(item.get("时长", 10.0))
            # 本场景要不要接上一段的续帧，由这一行自己的标记决定
            启用场景续接 = scene_relay_flag(item)

            print(f"✅ [凤希AI] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在执行第 {行索引 + 1}/{total_lines} 个场景，本场景 {时长} 秒")
            broadcast("scene_executing", {
                "current": 行索引 + 1,
                "total": total_lines,
                "duration": 时长,
                "message": f"正在执行第 {行索引 + 1}/{total_lines} 个场景，本场景 {时长} 秒"
            })

            path_list = [p.strip() for p in str(素材).split(",") if p.strip()]
            x = max(5, round(时长 * 24))
            frame = x + (5 - x) % 17

        except Exception as e:
            print(f"✅ [凤希AI] 场景加载异常：{e}")
            x = max(5, int(round(时长 * 24)))
            frame = x + (5 - x) % 17
            return (
                f"{通用提示词}{台词 or ''}{尾部通用提示词}",
                images,
                frame,
                参考音色,
                音频,
                启用场景续接,
            )

        # 按图片顺序查出角色声音（有声音的才输出，顺序对应前端 <Audio N>）
        characters = get_characters_by_avatars(path_list)

        for rel_path in path_list:
            parts = rel_path.split("/", 1)
            if len(parts) != 2:
                continue

            subdir, filename = parts
            full_dir = get_image_dir(subdir)
            full_path = os.path.join(full_dir, filename)

            if not os.path.exists(full_path):
                print(f"[凤希AI] 图片不存在：{full_path}")
                continue

            try:
                tensor = load_single_image(full_path)
                images.append(tensor)
            except Exception as e:
                print(f"[凤希AI] 加载失败：{full_path} => {e}")

            # 该图片对应的角色若有声音：参考音色截前 2 秒，音频输出全量
            char = characters.get(rel_path)
            if char and char.get("voice"):
                try:
                    audio = load_audio_tensor_from_file(char["voice"])
                    音频.append(audio)
                    waveform = audio["waveform"]
                    sample_rate = audio["sample_rate"]
                    cut_seconds = 音色剪切时长 if 音色剪切时长 > 0 else 时长
                    cut_frames = int((cut_seconds + 0.8) * sample_rate)
                    if waveform.size(-1) > cut_frames:
                        audio = slice_audio(audio, 0, cut_frames)
                    参考音色.append(audio)
                except Exception as e:
                    print(f"[凤希AI] 音频加载失败：{char['voice']} => {e}")

        return (
            提示词,
            images,
            frame,
            参考音色,
            音频,
            启用场景续接,
        )


class FxAiScreenLoadSimple:
    """简化版场景加载：只取台词+图片，循环取值（取余数），去掉音频和帧数。"""

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

    RETURN_TYPES = ("STRING", "IMAGE", "BOOLEAN")
    RETURN_NAMES = ("台词", "素材", "场景续接")

    FUNCTION = "get_scene_data"
    CATEGORY = "凤希AI/影视剧场"

    def get_scene_data(self, 场景数据, 行索引, 通用提示词="", 尾部通用提示词=""):
        台词 = ""
        提示词 = ""
        images = []
        启用场景续接 = False

        try:
            total_lines = len(场景数据) if isinstance(场景数据, list) else 0

            if total_lines == 0:
                raise ValueError("场景数据为空")

            实际索引 = 行索引 % total_lines
            item = 场景数据[实际索引]
            台词 = item.get("台词", item.get("提示词文本", ""))
            # 当前场景台词为空时，回退到第一个场景的台词
            if not 台词.strip() and total_lines > 0:
                first_item = 场景数据[0]
                台词 = first_item.get("台词", first_item.get("提示词文本", ""))
            提示词 = 通用提示词 + 台词 + 尾部通用提示词
            素材 = item.get("素材", "")
            # 循环场景时按「实际落到的那一行」取续接标记，跟取台词/素材同一行
            启用场景续接 = scene_relay_flag(item)

            print(f"✅ [凤希AI] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在执行第 {行索引 + 1}/{total_lines} 个场景（循环取值：实际场景={实际索引 + 1}）")
            broadcast("scene_executing", {
                "current": 行索引 + 1,
                "total": total_lines,
                "actual_index": 实际索引,
                "message": f"正在执行第 {行索引 + 1}/{total_lines} 个场景"
            })

            path_list = [p.strip() for p in str(素材).split(",") if p.strip()]

        except Exception as e:
            print(f"✅ [凤希AI] 场景加载异常：{e}")
            return (提示词, images, 启用场景续接)

        for rel_path in path_list:
            parts = rel_path.split("/", 1)
            if len(parts) != 2:
                continue

            subdir, filename = parts
            full_dir = get_image_dir(subdir)
            full_path = os.path.join(full_dir, filename)

            if not os.path.exists(full_path):
                print(f"[凤希AI] 图片不存在：{full_path}")
                continue

            try:
                tensor = load_single_image(full_path)
                images.append(tensor)
            except Exception as e:
                print(f"[凤希AI] 加载失败：{full_path} => {e}")

        return (提示词, images, 启用场景续接)