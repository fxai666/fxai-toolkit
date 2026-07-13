import os
import json
import re
import folder_paths
import server
import torch
import safetensors
from aiohttp import web
from nodes import LoraLoader

# 新增类型转换函数
def safe_float(val, default=1.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

# ====================== LoRA 触发词提取 ======================
def extract_lora_trigger_words(lora_path):
    try:
        ext = os.path.splitext(lora_path)[1].lower()
        if ext == ".safetensors":
            with safetensors.safe_open(lora_path, framework="pt") as f:
                meta = f.metadata()
        elif ext in (".bin", ".ckpt"):
            ckpt = torch.load(lora_path, map_location="cpu", weights_only=True)
            meta = ckpt.get("metadata", {})
        else:
            return []

        tags = []
        prefix = meta.get("ss_caption_prefix", "").strip()
        if prefix:
            tags.append(prefix)
        
        freq = meta.get("ss_tag_frequency", "")
        if freq and not prefix:
            try:
                freq_dict = json.loads(freq)
                top_tags = [k for k, v in freq_dict.items() if v >= 5]
                tags.extend(top_tags[:5])
            except Exception:
                pass

        return tags
    except Exception:
        return []

# ====================== 工具函数 ======================
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

def get_config_path(lora_name):
    cfg_dir = get_lora_config_dir()
    pure_name = os.path.splitext(clean_filename(lora_name))[0]
    return safe_path_join(cfg_dir, f"{pure_name}.json")

def load_config(lora_name):
    path = get_config_path(lora_name)
    default_config = {
        "enabled": True,
        "model_strength": 1.0,
        "clip_strength": -1.0,
        "trigger_words": [],
        "invert": False,
        "fade_start": 1.0,
        "fade_end": 1.0
    }

    # 1. 有配置文件 → 直接加载
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
                if "model_strength" in user_config:
                    user_config["model_strength"] = safe_float(user_config["model_strength"], 1.0)
                if "clip_strength" in user_config:
                    user_config["clip_strength"] = safe_float(user_config["clip_strength"], -1.0)
                if "fade_start" in user_config:
                    user_config["fade_start"] = safe_float(user_config["fade_start"], 1.0)
                if "fade_end" in user_config:
                    user_config["fade_end"] = safe_float(user_config["fade_end"], 1.0)
                default_config.update(user_config)
        except Exception:
            pass

    lora_path = folder_paths.get_full_path("loras", lora_name)
    if lora_path:
        triggers = extract_lora_trigger_words(lora_path)
        default_config["trigger_words"] = triggers

    if path:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return default_config

# ====================== API 接口 ======================
async def api_get_lora_files(request):
    loras = folder_paths.get_filename_list("loras")
    loras_sorted = sorted(loras, key=lambda x: x.lower())
    result = {}
    for lora_name in loras_sorted:
        result[lora_name] = load_config(lora_name)
    return web.json_response(result)

async def api_get_lora_config(request):
    name = request.query.get("name", "")
    return web.json_response(load_config(name))

server.PromptServer.instance.routes.get("/fxai/lora/files")(api_get_lora_files)
server.PromptServer.instance.routes.get("/fxai/lora/config")(api_get_lora_config)

# ====================== 主节点 ======================
class FxAiLoraLoader:
    CATEGORY = "凤希AI/LoRA"
    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "提示词")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "clip": ("CLIP",),
                "提示词": ("STRING", {"forceInput": True}),
                "lora_data": ("STRING", {"default": "[]", "multiline": True}),
            }
        }

    def run(self, model, clip=None, 提示词=None, lora_data="[]"):
        # 原始输入前置提示词
        base_prompt = 提示词.strip() if (提示词 is not None and 提示词.strip()) else ""
        lora_tag_lines = []

        try:
            items = json.loads(lora_data)
        except Exception:
            items = []

        # 严格按照 lora_data 数组顺序遍历，顺序完全保留
        for item in items:
            lora_name = item.get("lora_name", "")
            enabled = item.get("enabled", True)
            model_str = safe_float(item.get("model_strength", 1.0))
            clip_str = safe_float(item.get("clip_strength", -1.0))
            invert = item.get("invert", False)
            fade_start = safe_float(item.get("fade_start", 1.0))
            fade_end = safe_float(item.get("fade_end", 1.0))
            trigger_words = item.get("trigger_words", [])

            if int(float(clip_str)) == -1:
                clip_str = model_str

            if not enabled or not lora_name:
                continue

            if invert:
                model_str = -model_str
                clip_str = -clip_str
            model_str *= fade_start
            clip_str *= fade_end

            try:
                lora_path = folder_paths.get_full_path("loras", lora_name)
                if lora_path is None or not os.path.exists(lora_path):
                    raise FileNotFoundError(f"LoRA 文件不存在：{lora_name}，请在以下网盘进行下载\n夸克：https://pan.quark.cn/s/a1213641f8c2#/list/share/5e7665b0a764d4d9a9adbd2ec4b71f7\n迅雷：https://pan.xunlei.com/s/VOm0BR-N1g-SuFr_RceCmVKzA1?pwd=qnc3")
                
                model, clip = LoraLoader().load_lora(
                    model, clip, lora_name, model_str, clip_str
                )
            except Exception as e:
                raise RuntimeError(f"[凤希AI LoRA加载失败] {lora_name}\n失败原因：{str(e)}") from e

            # 当前LoRA的tag拼接成一行，加入列表（顺序和加载LoRA完全一致）
            if isinstance(trigger_words, list):
                tag_list = [t.strip() for t in trigger_words if t.strip()]
            else:
                tag_list = [str(trigger_words).strip()] if str(trigger_words).strip() else []
            
            if tag_list:
                line_text = ", ".join(tag_list)
                lora_tag_lines.append(line_text)

        final_prompt = f'{base_prompt}\n' + ",".join(lora_tag_lines)
        return (model, clip, final_prompt)