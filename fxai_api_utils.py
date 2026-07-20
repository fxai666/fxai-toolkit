import os
import re
import json
import logging
import folder_paths
from aiohttp import web
from server import PromptServer

app = PromptServer.instance.app

# ===================== CORS中间件（绕过 ComfyUI 的 origin_only 检查） =====================
@web.middleware
async def fxai_cors_middleware(request, handler):
    if not request.path.startswith("/fxai/"):
        return await handler(request)

    # OPTIONS 预检请求直接返回
    if request.method == "OPTIONS":
        return web.Response(headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization"
        })

    # 直接调用路由处理器，跳过 ComfyUI 的 origin_only 等中间件
    try:
        route_handler = request._match_info.handler
        resp = await route_handler(request)
    except web.HTTPException as exc:
        resp = exc
    except Exception as e:
        print(f"[fxai] 路由处理失败({request.path}): {e}，退回到中间件链")
        resp = await handler(request)

    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return resp

# 注入到中间件链首位
app._middlewares.insert(0, fxai_cors_middleware)

# ===================== 健康检测 =====================
async def health_check(request):
    return web.json_response({
        "status": "ok",
        "service": "fxai-toolkit",
        "version": "1.0"
    })

def get_folder(request):
    subdir = request.query.get("subdir", "")

    comfy_root = folder_paths.base_path
    safe_sub = re.sub(r'[\\/*?:"<>|]', "", subdir.strip())
    safe_sub = os.path.normpath(safe_sub).lstrip(".")
    target_dir = os.path.join(comfy_root, "fxai", safe_sub)
    target_dir = os.path.abspath(target_dir)
    allowed_base = os.path.abspath(os.path.join(comfy_root, "fxai"))
    if not target_dir.startswith(allowed_base):
        return web.json_response({"root_path": allowed_base, "sub_dirs": []})

    dir_names = []
    if os.path.isdir(target_dir):
        for name in os.listdir(target_dir):
            full_path = os.path.join(target_dir, name)
            if os.path.isdir(full_path):
                dir_names.append(name)
    sorted_dirs = sorted(dir_names)

    return web.json_response({
        "root_path": target_dir,
        "sub_dirs": sorted_dirs
    })

# ===================== 文件类型分类 =====================
EXT_IMAGE = {'.png','.jpg','.jpeg','.gif','.bmp','.webp','.tiff','.tif','.ico','.svg'}
EXT_AUDIO = {'.mp3','.wav','.ogg','.flac','.aac','.m4a','.wma','.opus'}
EXT_VIDEO = {'.mp4','.avi','.mov','.mkv','.webm','.flv','.wmv','.ts','.mts'}
EXT_TEXT  = {'.txt','.json','.xml','.csv','.yaml','.yml','.md','.log','.cfg','.ini'}

def categorize_file(name, full_path):
    ext = os.path.splitext(name)[1].lower()
    try:
        size = os.path.getsize(full_path)
    except:
        size = 0
    try:
        mtime = os.path.getmtime(full_path)
    except:
        mtime = 0
    item = {"name": name, "size": size, "mtime": mtime}
    if ext in EXT_IMAGE:
        return ("images", item)
    if ext in EXT_AUDIO:
        return ("audio", item)
    if ext in EXT_VIDEO:
        return ("video", item)
    if ext in EXT_TEXT:
        return ("text", item)
    return ("other", item)

# ===================== 统一文件列表 API =====================
async def list_files(request):
    dir_key = request.query.get("dir", "fxai")
    subdir = request.query.get("subdir", "")
    target = resolve_target(dir_key, subdir)

    result = {"images": [], "audio": [], "video": [], "text": [], "other": []}
    if os.path.isdir(target):
        for name in sorted(os.listdir(target), key=str.lower):
            full = os.path.join(target, name)
            if os.path.isfile(full) and not name.startswith('.'):
                cat, item = categorize_file(name, full)
                result[cat].append(item)

    return web.json_response(result)

async def list_subdirs(request):
    dir_key = request.query.get("dir", "fxai")
    subdir = request.query.get("subdir", "")
    target = resolve_target(dir_key, subdir)

    dirs = []
    if os.path.isdir(target):
        for name in os.listdir(target):
            full = os.path.join(target, name)
            if os.path.isdir(full) and not name.startswith('.'):
                try:
                    ctime = os.path.getctime(full)
                except:
                    ctime = 0
                dirs.append({"name": name, "ctime": int(ctime)})
        dirs.sort(key=lambda d: d["ctime"], reverse=True)

    return web.json_response({"dir": target, "subdirs": dirs})

# ===================== 解析目标路径 =====================
def resolve_target(dir_key, subdir=""):
    if dir_key == "input":
        root = folder_paths.get_input_directory()
    elif dir_key == "output":
        root = folder_paths.get_output_directory()
    else:
        root = os.path.join(folder_paths.base_path, "fxai")
        if dir_key and dir_key != "fxai":
            subdir = os.path.join(dir_key, subdir) if subdir else dir_key
    target = os.path.join(root, subdir) if subdir else root
    return os.path.abspath(target)

# ===================== 删除文件 API =====================
async def delete_file(request):
    dir_key = request.query.get("dir", "fxai")
    subdir = request.query.get("subdir", "")
    filename = request.query.get("filename", "")

    if not filename:
        return web.json_response({"error": "缺少 filename"}, status=400)
    filename = os.path.basename(filename)  # 防路径穿越
    target = resolve_target(dir_key, subdir)
    filepath = os.path.join(target, filename)

    if not os.path.isfile(filepath):
        return web.json_response({"error": "文件不存在"}, status=404)

    try:
        os.remove(filepath)
        return web.json_response({"success": True, "deleted": filename})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

# ===================== 删除目录 API（递归） =====================
async def delete_folder(request):
    dir_key = request.query.get("dir", "fxai")
    subdir = request.query.get("subdir", "")
    folder_name = request.query.get("folder", "")

    if not folder_name:
        return web.json_response({"error": "缺少 folder"}, status=400)
    folder_name = os.path.basename(folder_name)  # 防路径穿越
    target = resolve_target(dir_key, subdir)
    folder_path = os.path.join(target, folder_name)

    if not os.path.isdir(folder_path):
        return web.json_response({"error": "目录不存在"}, status=404)

    # 安全校验：确保不超出允许范围
    if dir_key == "input":
        allowed = os.path.abspath(folder_paths.get_input_directory())
    elif dir_key == "output":
        allowed = os.path.abspath(folder_paths.get_output_directory())
    else:
        allowed = os.path.abspath(os.path.join(folder_paths.base_path, "fxai"))

    if not folder_path.startswith(allowed):
        return web.json_response({"error": "禁止删除此目录"}, status=403)

    try:
        import shutil
        shutil.rmtree(folder_path)
        return web.json_response({"success": True, "deleted": folder_name})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

# ===================== 代理 input/output 文件预览（绕过 ComfyUI origin 检查） =====================
async def io_preview(request):
    filename = request.query.get("filename", "")
    dir_type = request.query.get("type", "input")
    subfolder = request.query.get("subfolder", "")

    if not filename:
        return web.json_response({"error": "缺少 filename"}, status=400)
    filename = os.path.basename(filename)

    if dir_type == "input":
        root = folder_paths.get_input_directory()
    elif dir_type == "output":
        root = folder_paths.get_output_directory()
    else:
        return web.json_response({"error": "无效的 type"}, status=400)

    filepath = os.path.join(root, subfolder, filename)
    filepath = os.path.abspath(filepath)

    if not os.path.isfile(filepath):
        return web.json_response({"error": "文件不存在"}, status=404)

    return web.FileResponse(filepath)

# ===================== 文本预览（从 fxai/prompts 读取） =====================
async def text_preview(request):
    filename = request.query.get("filename", "")
    subdir = request.query.get("subdir", "")

    if not filename:
        return web.json_response({"error": "缺少 filename"}, status=400)
    filename = os.path.basename(filename)

    root = os.path.join(folder_paths.base_path, "fxai", "prompts")
    target = os.path.join(root, subdir, filename) if subdir else os.path.join(root, filename)
    target = os.path.abspath(target)

    allowed = os.path.abspath(os.path.join(folder_paths.base_path, "fxai", "prompts"))
    if not target.startswith(allowed):
        return web.json_response({"error": "禁止访问"}, status=403)

    if not os.path.isfile(target):
        return web.json_response({"error": "文件不存在"}, status=404)

    return web.FileResponse(target)

# ===================== 模型目录浏览 =====================
async def list_model_subdirs(request):
    root = folder_paths.models_dir
    dirs = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root), key=str.lower):
            full = os.path.join(root, name)
            if os.path.isdir(full) and not name.startswith('.'):
                try:
                    count = len([f for f in os.listdir(full) if os.path.isfile(os.path.join(full, f)) and not f.startswith('.')])
                except:
                    count = 0
                dirs.append({"name": name, "file_count": count})
    return web.json_response({"root": root, "subdirs": dirs})

async def list_model_files(request):
    model_type = request.query.get("type", "")
    if not model_type:
        return web.json_response({"error": "缺少 type"}, status=400)
    model_type = os.path.basename(model_type)
    root = os.path.join(folder_paths.models_dir, model_type)
    files = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root), key=str.lower):
            full = os.path.join(root, name)
            if os.path.isfile(full) and not name.startswith('.'):
                ext = os.path.splitext(name)[1].lower()
                try:
                    size = os.path.getsize(full)
                except:
                    size = 0
                try:
                    mtime = os.path.getmtime(full)
                except:
                    mtime = 0
                files.append({"name": name, "ext": ext, "size": size, "mtime": int(mtime)})
    return web.json_response({"type": model_type, "path": root, "files": files})

# ===================== Workflows 目录（完整工作流） =====================
WORKFLOWS_DIR = os.path.join(os.path.dirname(__file__), "workflows")

async def list_workflows(request):
    files = []
    if os.path.isdir(WORKFLOWS_DIR):
        for name in sorted(os.listdir(WORKFLOWS_DIR), key=str.lower):
            if name.endswith(".json"):
                files.append({"filename": name, "title": name.replace(".json", "")})
    return web.json_response({"workflows": files})

# ===================== Workflows API 目录（API Format） =====================
WORKFLOWS_API_DIR = os.path.join(os.path.dirname(__file__), "workflows", "api")
if not os.path.isdir(WORKFLOWS_API_DIR):
    try: os.makedirs(WORKFLOWS_API_DIR)
    except: pass

try:
    PromptServer.instance.routes.get("/fxai/health")(health_check)
    PromptServer.instance.routes.get("/fxai/folder/list")(get_folder)
    PromptServer.instance.routes.get("/fxai/files/list")(list_files)
    PromptServer.instance.routes.get("/fxai/files/subdirs")(list_subdirs)
    PromptServer.instance.routes.post("/fxai/files/delete")(delete_file)
    PromptServer.instance.routes.post("/fxai/folders/delete")(delete_folder)
    PromptServer.instance.routes.get("/fxai/io/view")(io_preview)
    PromptServer.instance.routes.get("/fxai/text/preview")(text_preview)
    PromptServer.instance.routes.get("/fxai/workflows/list")(list_workflows)
    PromptServer.instance.routes.get("/fxai/models/subdirs")(list_model_subdirs)
    PromptServer.instance.routes.get("/fxai/models/files")(list_model_files)
except Exception as e:
    print(f"❌ fxai API 挂载失败：{e}")
