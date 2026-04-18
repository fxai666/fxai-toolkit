import os
import re
import folder_paths
import server
from aiohttp import web
import mimetypes

# 安全路径校验
def safe_path_join(base_dir, path):
    base_dir = os.path.abspath(base_dir)
    full_path = os.path.abspath(os.path.join(base_dir, path))
    if not full_path.startswith(base_dir):
        return None
    return full_path

# 提示词目录
def get_prompt_dir(subdir=""):
    comfy_root = folder_paths.base_path
    base_dir = "fxai/prompts"
    target_dir = os.path.join(comfy_root, base_dir)
    if subdir:
        subdir = re.sub(r'[\\/*?:"<>|]', "", subdir)
        target_dir = os.path.join(target_dir, subdir)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

# 列出 txt 文件（直接显示原文件名，不排序、不编号）
def list_prompts(target_dir):
    if not os.path.isdir(target_dir):
        return []
    files = []
    for f in os.listdir(target_dir):
        fp = os.path.join(target_dir, f)
        if os.path.isfile(fp) and f.lower().endswith(".txt"):
            files.append(f)
    files.sort()  # 按字母顺序展示
    return files

# ---------- API ----------
async def get_preview(request):
    subdir = request.query.get("subdir", "")
    filename = request.query.get("filename", "")
    if not filename:
        return web.json_response({"error": "未提供文件名"}, status=400)
    target_dir = get_prompt_dir(subdir)
    safe_file = safe_path_join(target_dir, filename)
    if not safe_file or not os.path.exists(safe_file):
        return web.json_response({"error": "文件不存在"}, status=404)
    try:
        with open(safe_file, "r", encoding="utf-8") as f:
            return web.Response(text=f.read(), content_type="text/plain; charset=utf-8")
    except:
        return web.json_response({"error": "读取失败"}, status=500)

async def get_file_list(request):
    subdir = request.query.get("subdir", "")
    target_dir = get_prompt_dir(subdir)
    files = list_prompts(target_dir)
    return web.json_response({"files": files, "total": len(files)})

# 上传：直接用前端传的文件名，不自动编号
async def upload_prompt_custom(request):
    try:
        data = await request.post()
        prompt_file = data.get("prompt")
        subdir = data.get("subdir", "")
        if not prompt_file or not hasattr(prompt_file, "file"):
            return web.json_response({"error": "无有效文件"}, status=400)

        filename = prompt_file.filename
        filename = re.sub(r'[\\/*?:"<>|]', "", filename)
        if not filename.lower().endswith(".txt"):
            filename = os.path.splitext(filename)[0] + ".txt"

        target_dir = get_prompt_dir(subdir)
        save_path = safe_path_join(target_dir, filename)
        if not save_path:
            return web.json_response({"error": "非法路径"}, status=403)

        with open(save_path, "wb") as f:
            f.write(prompt_file.file.read())
        return web.json_response({"success": True, "name": filename})
    except Exception as e:
        return web.json_response({"error": f"上传失败：{str(e)}"}, status=500)

# 废弃接口，保留兼容
async def get_next_number(request):
    return web.json_response({"next_num": 0})

async def apply_changes(request):
    return web.json_response({"success": True, "files": []})

# 注册路由
try:
    server.PromptServer.instance.routes.get("/fxpromptmanager/preview")(get_preview)
    server.PromptServer.instance.routes.get("/fxpromptmanager/list")(get_file_list)
    server.PromptServer.instance.routes.post("/fxpromptmanager/upload")(upload_prompt_custom)
    server.PromptServer.instance.routes.get("/fxpromptmanager/next_number")(get_next_number)
    server.PromptServer.instance.routes.post("/fxpromptmanager/apply")(apply_changes)
    print("✅ 凤希提示词管理器 已加载")
except Exception as e:
    print(f"❌ 提示词管理器加载失败：{e}")

# ComfyUI 节点
class FxAiPromptManager:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "目录": ("STRING", {"default": "default"}),
            },
            "optional": {
                "手动输入提示词": ("STRING", {"multiline": True, "default": ""}),
                "保存文件名": ("STRING", {"default": "my_prompt"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("文件列表", "目录路径", "总数")
    FUNCTION = "run"
    CATEGORY = "凤希AI"

    def save_manual_prompt(self, text, save_dir, filename):
        if not text.strip():
            return
        try:
            filename = re.sub(r'[\\/*?:"<>|]', "", filename)
            if not filename.lower().endswith(".txt"):
                filename += ".txt"
            save_path = os.path.join(save_dir, filename)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(text.strip())
            print(f"[凤希] 已保存：{save_path}")
        except:
            pass

    def run(self, 目录="", 手动输入提示词="", 保存文件名="my_prompt"):
        target_dir = get_prompt_dir(目录)
        if 手动输入提示词.strip():
            self.save_manual_prompt(手动输入提示词, target_dir, 保存文件名)
        files = list_prompts(target_dir)
        file_str = "\n".join(files) if files else "无文件"
        return (file_str, target_dir, len(files))