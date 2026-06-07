import os
import json
import re
import folder_paths
import server
from aiohttp import web
from nodes import LoraLoader

# ====================== 工具函数（保留配置文件系统）======================
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
    default = {
        "enabled": True,
        "model_strength": 1.0,
        "clip_strength": 1.0,
        "trigger_words": [],
        "invert": False,
        "fade_start": 1.0,
        "fade_end": 1.0
    }
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

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
    RETURN_NAMES = ("MODEL", "CLIP", "触发词")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "clip": ("CLIP",),
                "lora_data": ("STRING", {"default": "[]", "multiline": True}),
            }
        }

    def run(self, model, clip=None, lora_data="[]"):
        triggers = []
        try:
            items = json.loads(lora_data)
        except:
            items = []

        for item in items:
            lora_name = item.get("lora_name", "")
            enabled = item.get("enabled", True)
            model_str = item.get("model_strength", 1.0)
            clip_str = item.get("clip_strength", 1.0)
            invert = item.get("invert", False)
            fade_start = item.get("fade_start", 1.0)
            fade_end = item.get("fade_end", 1.0)
            trigger_words = item.get("trigger_words", [])

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
                    raise FileNotFoundError(f"LoRA 文件不存在：{lora_name}，请在以下网盘进行下载\n夸克：https://pan.quark.cn/s/a1213641f8c2#/list/share/5e75665b0a764d4d9a9adbd2ec4b71f7\n迅雷：https://pan.xunlei.com/s/VOm0BR-N1g-SuFr_RceCmVKzA1?pwd=qnc3")
                
                model, clip = LoraLoader().load_lora(
                    model, clip, lora_name, model_str, clip_str
                )
            except Exception as e:
                raise RuntimeError(f"[凤希AI LoRA加载失败] {lora_name}\n失败原因：{str(e)}") from e

            if isinstance(trigger_words, list):
                triggers.extend(trigger_words)
            else:
                triggers.append(str(trigger_words))

        triggers_clean = list(set([t.strip() for t in triggers if t.strip()]))
        trigger_out = ", ".join(triggers_clean)
        return (model, clip, trigger_out)