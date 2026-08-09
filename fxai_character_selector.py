# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# 选择角色：通过资源图片选择器选一张角色形象照，去短剧角色库（fxai.db）
# 按头像路径匹配角色，命中且配置了音色时输出该角色音频。

import os
import re
import torch
import folder_paths
from fxai_image_utils import load_single_image
from fxai_audio_utils import load_audio_tensor_from_file, slice_audio
from fxai_character_profile_manager import get_characters_by_avatars


def get_image_path(full_relative_path):
    comfy_root = folder_paths.base_path
    base_dir = "fxai/image"
    target_path = os.path.join(comfy_root, base_dir, full_relative_path)
    target_path = os.path.abspath(target_path)
    allowed_base = os.path.abspath(os.path.join(comfy_root, base_dir))
    if not target_path.startswith(allowed_base):
        raise ValueError("路径穿越被拒绝")
    return target_path


class FxAiCharacterSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "角色头像": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "layout": "hidden"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("图片", "音频", "名称", "描述")
    FUNCTION = "select"
    CATEGORY = "凤希AI/角色"

    def select(self, 角色头像):
        avatar = (角色头像 or "").strip()
        audio = None
        角色名 = ""
        角色描述 = ""
        头像图片 = None

        if avatar:
            try:
                characters = get_characters_by_avatars([avatar])
                char = characters.get(avatar)
                if char:
                    角色名 = char.get("name") or ""
                    角色描述 = char.get("description") or ""
                    if char.get("voice"):
                        audio = load_audio_tensor_from_file(char["voice"])
                        cut_frames = int(3.0 * audio["sample_rate"])
                        if audio["waveform"].size(-1) > cut_frames:
                            audio = slice_audio(audio, 0, cut_frames)
            except Exception as e:
                print(f"[凤希AI选择角色] 查询角色音频失败：{e}")

            try:
                if os.path.exists(get_image_path(avatar)):
                    头像图片 = load_single_image(get_image_path(avatar))
            except Exception:
                pass

        return (头像图片, audio, 角色名, 角色描述)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
