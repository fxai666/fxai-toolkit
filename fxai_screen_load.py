import os
import re
import math
import folder_paths
from fxai_image_utils import load_single_image
from fxai_audio_utils import load_audio_tensor_from_file, slice_audio
from fxai_character_profile_manager import get_characters_by_avatars

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

    RETURN_TYPES = ("STRING", "IMAGE", "INT", "LIST", "LIST")
    RETURN_NAMES = ("台词", "素材", "帧数", "参考音色", "音频")

    FUNCTION = "get_scene_data"
    CATEGORY = "凤希AI/影视剧场"

    def get_scene_data(self, 场景数据, 行索引, 通用提示词="", 尾部通用提示词=""):
        台词 = ""
        提示词 = ""
        时长 = 15.0
        frame = 0
        images = []
        参考音色 = []
        音频 = []

        try:
            total_lines = len(场景数据) if isinstance(场景数据, list) else 0

            if 行索引 < 0 or total_lines == 0 or 行索引 >= total_lines:
                raise IndexError(f"行索引越界：{行索引} / 总行数：{total_lines}")

            item = 场景数据[行索引]
            台词 = item.get("台词", item.get("提示词文本", ""))
            提示词 = 通用提示词 + 台词 + 尾部通用提示词
            素材 = item.get("素材", "")
            时长 = float(item.get("时长", 10.0))

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
                参考音色,
                音频,
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
                print(f"[凤希] 图片不存在：{full_path}")
                continue

            try:
                tensor = load_single_image(full_path)
                images.append(tensor)
            except Exception as e:
                print(f"[凤希] 加载失败：{full_path} => {e}")

            # 该图片对应的角色若有声音：参考音色截前 2 秒，音频输出全量
            char = characters.get(rel_path)
            if char and char.get("voice"):
                try:
                    audio = load_audio_tensor_from_file(char["voice"])
                    音频.append(audio)
                    waveform = audio["waveform"]
                    sample_rate = audio["sample_rate"]
                    cut_frames = int(时长 * sample_rate)
                    if waveform.size(-1) > cut_frames:
                        audio = slice_audio(audio, 0, cut_frames)
                    参考音色.append(audio)
                except Exception as e:
                    print(f"[凤希] 音频加载失败：{char['voice']} => {e}")

        return (
            提示词,
            images,
            frame,
            参考音色,
            音频,
        )