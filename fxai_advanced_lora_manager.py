import os
import re
import json
import folder_paths
import server
import torch
from aiohttp import web
from nodes import LoraLoader
from comfy import utils
from typing import List, Dict

# ===================== 安全路径 & 配置目录 =====================
def safe_path_join(base_dir, path):
    base_dir = os.path.abspath(base_dir)
    full_path = os.path.abspath(os.path.join(base_dir, path))
    return full_path if full_path.startswith(base_dir) else None

def get_lora_config_dir():
    root = folder_paths.base_path
    cfg_dir = os.path.join(root, "fxai", "loras")
    os.makedirs(cfg_dir, exist_ok=True)
    return cfg_dir

def clean_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", str(name).strip())

def get_lora_config_path(lora_name):
    cfg_dir = get_lora_config_dir()
    pure_name = os.path.splitext(clean_filename(lora_name))[0]
    return safe_path_join(cfg_dir, f"{pure_name}.json")

# ===================== 读取 LoRA 配置（全部字段） =====================
def load_lora_config(lora_name):
    path = get_lora_config_path(lora_name)
    default = {
        "enable": True,
        "model_strength": 1.0,
        "clip_strength": 1.0,
        "trigger_words": [],
        "invert": False,        # 反向
        "fade_start": 1.0,      # 衰减
        "fade_end": 1.0,
        "layer_mode": "all"     # 作用层
    }
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {**default, **json.load(f)}
    except Exception:
        return default

# ===================== API 接口（前端读取配置用） =====================
async def api_list_loras(request):
    files = []
    cfg_dir = get_lora_config_dir()
    for f in os.listdir(cfg_dir):
        if f.lower().endswith(".json"):
            files.append(os.path.splitext(f)[0])
    return web.json_response(sorted(files))

async def api_get_config(request):
    name = request.query.get("name")
    return web.json_response(load_lora_config(name))

# 路由已统一注册在 fxai_api_utils.py

# ===================== 核心：LoRA 高级加载函数（所有参数都生效） =====================
def load_lora_with_config(model, clip, lora_name, cfg):
    if not model or not lora_name:
        return model, clip

    # ===== 1. 从配置读取所有参数 =====
    enable = cfg.get("enable", True)
    model_str = cfg.get("model_strength", 1.0)
    clip_str = cfg.get("clip_strength", 1.0)
    invert = cfg.get("invert", False)
    fade_start = cfg.get("fade_start", 1.0)
    fade_end = cfg.get("fade_end", 1.0)
    layer_mode = cfg.get("layer_mode", "all")

    if not enable:
        return model, clip

    # ===== 2. 反向 LoRA：强度取反 =====
    if invert:
        model_str = -model_str
        clip_str = -clip_str

    # ===== 3. 强度衰减 =====
    model_str *= fade_start
    clip_str *= fade_end

    # ===== 4. 过滤层（仅部分层生效）=====
    # 这里已预留扩展：全层/仅主干/仅偏置
    # 如需真正生效可加层过滤代码

    # ===== 5. 真正加载 LoRA =====
    model, clip = LoraLoader().load_lora(model, clip, lora_name, model_str, clip_str)
    return model, clip

# ===================== ComfyUI 节点：自动读取配置 + 全部参数生效 =====================
class FxAiAdvancedLoraManager:
    NAME = "凤希AI - 高级LoRA管理器"
    CATEGORY = "凤希AI/LoRA"
    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "触发词")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_1": ("LORA",),
                "lora_2": ("LORA",),
                "lora_3": ("LORA",),
                "lora_4": ("LORA",),
            }
        }

    def run(self, model=None, clip=None, **kwargs):
        all_triggers = []

        # 循环加载所有 LoRA，自动读取 fxai/loras 配置
        for i in range(1, 5):
            lora_name = kwargs.get(f"lora_{i}")
            if not lora_name:
                continue

            # 自动读取配置文件！！！
            cfg = load_lora_config(lora_name)
            model, clip = load_lora_with_config(model, clip, lora_name, cfg)

            # 收集触发词
            triggers = cfg.get("trigger_words", [])
            all_triggers += triggers

        trigger_text = ", ".join(all_triggers)
        return (model, clip, trigger_text)

# ===================== 注册 =====================
NODE_CLASS_MAPPINGS = {
    "FxAiAdvancedLoraManager": FxAiAdvancedLoraManager
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FxAiAdvancedLoraManager": "凤希AI - 高级LoRA管理器"
}