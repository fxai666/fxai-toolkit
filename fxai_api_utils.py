import os
import re
import json
import logging
import asyncio
import aiohttp
import subprocess
import folder_paths
from aiohttp import web
from server import PromptServer

app = PromptServer.instance.app

# ===================== CORS中间件（绕过 ComfyUI 的 origin_only 检查） =====================
@web.middleware
async def fxai_cors_middleware(request, handler):
    if not request.path.startswith("/fxai/"):
        return await handler(request)

    if request.method == "OPTIONS":
        return web.Response(headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization"
        })

    try:
        route_handler = request._match_info.handler
        resp = await route_handler(request)
    except web.HTTPException as exc:
        resp = exc
    except Exception as e:
        print(f"[fxai] 路由处理失败({request.path}): {e}，退回到中间件链")
        resp = await handler(request)

    if not isinstance(resp, web.WebSocketResponse):
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

def get_audio_duration(filepath):
    try:
        if os.path.splitext(filepath)[1].lower() == '.wav':
            import wave
            with wave.open(filepath, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return round(frames / rate, 1) if rate > 0 else None
    except:
        pass
    return None

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
    if ext in EXT_AUDIO:
        item["duration"] = get_audio_duration(full_path)
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

    # 支持 JSON body 传参，也兼容 query
    try:
        body = await request.json()
        dir_key = body.get("dir", dir_key)
        subdir = body.get("subdir", subdir)
        filenames = body.get("filenames", [])
    except Exception:
        filenames = []
    single = request.query.get("filename", "")
    if single:
        filenames.append(single)

    if not filenames:
        return web.json_response({"error": "缺少 filenames"}, status=400)

    target = resolve_target(dir_key, subdir)
    deleted = []

    for name in filenames:
        name = os.path.basename(name)
        filepath = os.path.join(target, name)
        if os.path.isfile(filepath):
            try:
                os.remove(filepath)
                deleted.append(name)
            except Exception as e:
                print(f"[fxai] 删除失败 {name}: {e}")

    # ===================== 删除后重编号（仅一次） =====================
    if deleted:
        try:
            remaining = sorted([f for f in os.listdir(target) if os.path.isfile(os.path.join(target, f)) and not f.startswith('.')])
            if len(remaining) > 1:
                sample = remaining[0]
                is_numbered = re.match(r'^\d{3}_(.+)', os.path.splitext(sample)[0])
                is_pure_number = re.match(r'^\d{3}$', os.path.splitext(sample)[0])
                if is_pure_number:
                    for idx, f in enumerate(remaining):
                        ext = os.path.splitext(f)[1]
                        new_name = f"{idx:03d}{ext}"
                        if f != new_name:
                            os.rename(os.path.join(target, f), os.path.join(target, new_name))
                elif is_numbered:
                    tmp = []
                    for idx, f in enumerate(remaining):
                        _, ext = os.path.splitext(f)
                        m = re.match(r'^\d{3}_(.+)', os.path.splitext(f)[0])
                        name_part = m.group(1) if m else os.path.splitext(f)[0]
                        new_name = f"{idx:03d}_{name_part}{ext}"
                        tmp_name = f"_tmp_{os.urandom(4).hex()}"
                        os.rename(os.path.join(target, f), os.path.join(target, tmp_name))
                        tmp.append((tmp_name, new_name))
                    for tmp_name, new_name in tmp:
                        os.rename(os.path.join(target, tmp_name), os.path.join(target, new_name))
        except Exception as e:
            print(f"[fxai] 删除后重编号失败: {e}")

    return web.json_response({"success": True, "deleted": deleted})

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

# ===================== 创建目录 API =====================
async def create_folder(request):
    dir_key = request.query.get("dir", "fxai")
    subdir = request.query.get("subdir", "")
    folder_name = request.query.get("folder", "").strip()

    if not folder_name:
        return web.json_response({"error": "缺少 folder"}, status=400)
    folder_name = os.path.basename(folder_name)
    if not folder_name:
        return web.json_response({"error": "无效的目录名"}, status=400)
    target = resolve_target(dir_key, subdir)
    folder_path = os.path.join(target, folder_name)

    if os.path.exists(folder_path):
        return web.json_response({"error": "目录已存在"}, status=409)

    try:
        os.makedirs(folder_path)
        return web.json_response({"success": True, "created": folder_name})
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
                if count > 0:
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

# ===================== 检查模型文件是否存在 =====================
async def check_model_file(request):
    model_type = request.query.get("type", "")
    filename = request.query.get("file", "")
    if not model_type or not filename:
        return web.json_response({"exists": False, "error": "缺少参数"}, status=400)
    model_type = os.path.basename(model_type)
    filename = os.path.basename(filename)
    filepath = os.path.join(folder_paths.models_dir, model_type, filename)
    exists = os.path.isfile(filepath)
    return web.json_response({"exists": exists, "path": filepath if exists else ""})

# ===================== 批量检查模型文件 =====================
async def check_model_files_batch(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"results": {}, "error": "body_not_json"}, status=400)
    paths = body.get("paths", "")
    if not paths:
        return web.json_response({"results": {}})
    results = {}
    for item in paths.split(","):
        item = item.strip()
        if not item:
            continue
        filepath = os.path.join(folder_paths.models_dir, item)
        exists = os.path.isfile(filepath)
        results[item] = exists
    return web.json_response({"results": results})

# ===================== 删除模型文件 =====================
async def delete_model_file(request):
    data = await request.post()
    model_type = data.get("type", "")
    filename = data.get("filename", "")
    if not model_type or not filename:
        return web.json_response({"error": "缺少参数"}, status=400)
    model_type = os.path.basename(model_type)
    filename = os.path.basename(filename)
    filepath = os.path.join(folder_paths.models_dir, model_type, filename)
    if not os.path.isfile(filepath):
        return web.json_response({"error": "文件不存在"}, status=404)
    try:
        os.remove(filepath)
        return web.json_response({"success": True})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

# ===================== 清理空模型文件夹 =====================
def is_dir_empty_recursive(path):
    for entry in os.listdir(path):
        full = os.path.join(path, entry)
        if entry.startswith('.'):
            continue
        if os.path.isfile(full):
            return False
        if os.path.isdir(full) and not is_dir_empty_recursive(full):
            return False
    return True

def remove_empty_dirs(path, deleted):
    for entry in os.listdir(path):
        full = os.path.join(path, entry)
        if entry.startswith('.') or not os.path.isdir(full):
            continue
        remove_empty_dirs(full, deleted)
        if is_dir_empty_recursive(full):
            try:
                os.rmdir(full)
                deleted.append(os.path.relpath(full, folder_paths.models_dir))
            except:
                pass

async def clean_empty_model_dirs(request):
    root = folder_paths.models_dir
    deleted = []
    if os.path.isdir(root):
        remove_empty_dirs(root, deleted)
    return web.json_response({"success": True, "deleted": deleted})

# ===================== ComfyUI Prompt 代理（解决跨域） =====================
async def proxy_prompt(request):
    try:
        body = await request.json()
    except:
        return web.json_response({"error": "无效的JSON"}, status=400)
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post("http://127.0.0.1:8188/prompt", json=body, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                data = await resp.json()
                return web.json_response(data)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=502)

# ===================== WebSocket 代理（解决跨域） =====================
async def proxy_ws(request):
    ws_server = web.WebSocketResponse()
    await ws_server.prepare(request)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect("ws://127.0.0.1:8188/ws?" + (request.query_string or "")) as ws_client:
                async def forward_client_to_server():
                    async for msg in ws_server:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await ws_client.send_str(msg.data)
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            await ws_client.send_bytes(msg.data)
                        elif msg.type == aiohttp.WSMsgType.CLOSE:
                            break
                    await ws_client.close()
                async def forward_server_to_client():
                    async for msg in ws_client:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await ws_server.send_str(msg.data)
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            await ws_server.send_bytes(msg.data)
                        elif msg.type == aiohttp.WSMsgType.CLOSE:
                            break
                    await ws_server.close()
                import asyncio
                await asyncio.gather(forward_client_to_server(), forward_server_to_client())
    except Exception as e:
        print(f"[fxai] WebSocket 代理错误: {e}")
    return ws_server

# ===================== 中断任务代理 =====================
async def proxy_interrupt(request):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post("http://127.0.0.1:8188/interrupt", timeout=aiohttp.ClientTimeout(total=30)) as resp:
                return web.json_response({"success": resp.status == 200})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=502)

# ===================== 队列状态代理 =====================
async def proxy_queue(request):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get("http://127.0.0.1:8188/queue", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                return web.json_response(await resp.json())
        except Exception as e:
            return web.json_response({"error": str(e)}, status=502)

# ===================== 历史记录代理 =====================
async def proxy_history(request):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get("http://127.0.0.1:8188/history", timeout=aiohttp.ClientTimeout(total=30)) as resp:
                data = await resp.json()
                return web.json_response(data)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=502)

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

# ===================== 系统工作流目录（user/default/workflows） =====================
SYSTEM_WORKFLOWS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "user", "default", "workflows"))

async def list_workflow_dirs(request):
    dirs = []
    if os.path.isdir(SYSTEM_WORKFLOWS_DIR):
        for name in sorted(os.listdir(SYSTEM_WORKFLOWS_DIR), key=str.lower):
            full = os.path.join(SYSTEM_WORKFLOWS_DIR, name)
            if os.path.isdir(full) and not name.startswith('.'):
                count = len([f for f in os.listdir(full) if f.endswith('.json') and os.path.isfile(os.path.join(full, f))])
                dirs.append({"name": name, "count": count, "path": name})
        dirs.insert(0, {"name": "根目录", "count": len([f for f in os.listdir(SYSTEM_WORKFLOWS_DIR) if f.endswith('.json') and os.path.isfile(os.path.join(SYSTEM_WORKFLOWS_DIR, f))]), "path": ""})
    return web.json_response({"dirs": dirs})

async def list_workflow_files(request):
    subdir = request.query.get("dir", "")
    target = os.path.join(SYSTEM_WORKFLOWS_DIR, subdir) if subdir else SYSTEM_WORKFLOWS_DIR
    files = []
    if os.path.isdir(target):
        for name in sorted(os.listdir(target), key=str.lower):
            if name.endswith(".json") and os.path.isfile(os.path.join(target, name)):
                full = os.path.join(target, name)
                try:
                    size = os.path.getsize(full)
                    mtime = os.path.getmtime(full)
                except:
                    size = 0; mtime = 0
                files.append({"filename": name, "title": name.replace(".json", ""), "size": size, "mtime": int(mtime)})
    return web.json_response({"files": files, "dir": subdir or ""})

async def upload_workflow(request):
    reader = await request.multipart()
    field = await reader.next()
    if not field or field.name != "file":
        return web.json_response({"success": False, "error": "缺少文件字段"})
    filename = field.filename
    if not filename.endswith(".json"):
        return web.json_response({"success": False, "error": "仅支持 .json 文件"})
    subdir = request.query.get("dir", "")
    target_dir = os.path.join(SYSTEM_WORKFLOWS_DIR, subdir) if subdir else SYSTEM_WORKFLOWS_DIR
    if not os.path.isdir(target_dir):
        try: os.makedirs(target_dir)
        except: return web.json_response({"success": False, "error": "目录创建失败"})
    dest = os.path.join(target_dir, filename)
    size = 0
    with open(dest, "wb") as f:
        while True:
            chunk = await field.read_chunk()
            if not chunk: break
            f.write(chunk)
            size += len(chunk)
    return web.json_response({"success": True, "filename": filename, "size": size})

async def delete_workflow(request):
    filename = request.query.get("filename", "")
    subdir = request.query.get("dir", "")
    if not filename:
        return web.json_response({"success": False, "error": "缺少文件名"})
    target = os.path.join(SYSTEM_WORKFLOWS_DIR, subdir, filename) if subdir else os.path.join(SYSTEM_WORKFLOWS_DIR, filename)
    target = os.path.abspath(target)
    if not target.startswith(os.path.abspath(SYSTEM_WORKFLOWS_DIR)):
        return web.json_response({"success": False, "error": "路径不合法"})
    if os.path.isfile(target):
        try: os.remove(target); return web.json_response({"success": True})
        except Exception as e: return web.json_response({"success": False, "error": str(e)})
    return web.json_response({"success": False, "error": "文件不存在"})

async def view_workflow(request):
    filename = request.query.get("filename", "")
    subdir = request.query.get("dir", "")
    if not filename:
        return web.json_response({"error": "缺少文件名"})
    target = os.path.join(SYSTEM_WORKFLOWS_DIR, subdir, filename) if subdir else os.path.join(SYSTEM_WORKFLOWS_DIR, filename)
    target = os.path.abspath(target)
    if not target.startswith(os.path.abspath(SYSTEM_WORKFLOWS_DIR)):
        return web.json_response({"error": "路径不合法"})
    if os.path.isfile(target):
        with open(target, "r", encoding="utf-8") as f:
            content = f.read()
        return web.Response(text=content, content_type="application/json")
    return web.json_response({"error": "文件不存在"}, status=404)

async def shutdown_pc(request):
    try:
        data = await request.json()
        wait = int(data.get("wait_seconds", 60))
    except Exception:
        wait = 60
    wait = max(5, min(wait, 86400))
    import platform, subprocess
    sysos = platform.system()
    try:
        if sysos == "Windows":
            result = subprocess.run(["shutdown", "-s", "-t", str(wait)], capture_output=True, text=True)
            if result.returncode == 0:
                return web.json_response({"success": True, "message": f"已设置 {wait} 秒后关机"})
            return web.json_response({"success": False, "error": result.stderr.strip()}, status=500)
        elif sysos in ("Linux", "Darwin"):
            minutes = max(1, (wait + 59) // 60)
            result = subprocess.run(["sudo", "shutdown", "-h", f"+{minutes}"], capture_output=True, text=True)
            if result.returncode == 0:
                return web.json_response({"success": True, "message": f"已设置 {minutes} 分钟后关机"})
            return web.json_response({"success": False, "error": result.stderr.strip()}, status=500)
        else:
            return web.json_response({"success": False, "error": f"不支持的系统: {sysos}"}, status=400)
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)
try:
    PromptServer.instance.routes.get("/fxai/health")(health_check)
    PromptServer.instance.routes.get("/fxai/folder/list")(get_folder)
    PromptServer.instance.routes.get("/fxai/files/list")(list_files)
    PromptServer.instance.routes.get("/fxai/files/subdirs")(list_subdirs)
    PromptServer.instance.routes.post("/fxai/files/delete")(delete_file)
    PromptServer.instance.routes.post("/fxai/folders/delete")(delete_folder)
    PromptServer.instance.routes.post("/fxai/folders/create")(create_folder)
    PromptServer.instance.routes.get("/fxai/io/view")(io_preview)
    PromptServer.instance.routes.get("/fxai/text/preview")(text_preview)
    PromptServer.instance.routes.get("/fxai/workflows/list")(list_workflows)
    PromptServer.instance.routes.get("/fxai/workflows/dirs")(list_workflow_dirs)
    PromptServer.instance.routes.get("/fxai/workflows/files")(list_workflow_files)
    PromptServer.instance.routes.post("/fxai/workflows/upload")(upload_workflow)
    PromptServer.instance.routes.post("/fxai/workflows/delete")(delete_workflow)
    PromptServer.instance.routes.get("/fxai/workflows/view")(view_workflow)
    PromptServer.instance.routes.get("/fxai/models/subdirs")(list_model_subdirs)
    PromptServer.instance.routes.get("/fxai/models/files")(list_model_files)
    PromptServer.instance.routes.get("/fxai/models/check")(check_model_file)
    PromptServer.instance.routes.post("/fxai/models/check-batch")(check_model_files_batch)
    PromptServer.instance.routes.post("/fxai/models/delete")(delete_model_file)
    PromptServer.instance.routes.post("/fxai/models/clean-empty")(clean_empty_model_dirs)
    PromptServer.instance.routes.post("/fxai/prompt")(proxy_prompt)
    PromptServer.instance.routes.get("/fxai/ws")(proxy_ws)
    PromptServer.instance.routes.get("/fxai/history")(proxy_history)
    PromptServer.instance.routes.get("/fxai/queue")(proxy_queue)
    PromptServer.instance.routes.post("/fxai/interrupt")(proxy_interrupt)
    PromptServer.instance.routes.post("/fxai/shutdown")(shutdown_pc)
except Exception as e:
    print(f"❌ fxai API 挂载失败：{e}")
