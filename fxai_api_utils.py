import os
import re
import json
import math
import logging
import asyncio
import subprocess
import mimetypes
import folder_paths
import aiohttp
from aiohttp import web
from server import PromptServer
from datetime import datetime

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

app._middlewares.insert(0, fxai_cors_middleware)

# ===================== 通用工具函数 =====================

def _safe_path_join(base_dir, path):
    base_dir = os.path.abspath(base_dir)
    full_path = os.path.abspath(os.path.join(base_dir, path))
    return full_path if full_path.startswith(base_dir) else None

def _sanitize_filename(filename):
    name = re.sub(r'[\\/*?:"<>|]', '', filename)
    name = name.strip()
    return name

EXT_IMAGE = {'.png','.jpg','.jpeg','.gif','.bmp','.webp','.tiff','.tif','.ico','.svg'}
EXT_AUDIO = {'.mp3','.wav','.ogg','.flac','.aac','.m4a','.wma','.opus'}
EXT_VIDEO = {'.mp4','.avi','.mov','.mkv','.webm','.flv','.wmv','.ts','.mts'}
EXT_TEXT  = {'.txt','.json','.xml','.csv','.yaml','.yml','.md','.log','.cfg','.ini'}

def _get_audio_duration(filepath):
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

def _categorize_file(name, full_path):
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
        item["duration"] = _get_audio_duration(full_path)
    if ext in EXT_IMAGE:
        return ("images", item)
    if ext in EXT_AUDIO:
        return ("audio", item)
    if ext in EXT_VIDEO:
        return ("video", item)
    if ext in EXT_TEXT:
        return ("text", item)
    return ("other", item)

def _resolve_target(dir_key, subdir=""):
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

def _is_dir_empty_recursive(path):
    for entry in os.listdir(path):
        full = os.path.join(path, entry)
        if entry.startswith('.'):
            continue
        if os.path.isfile(full):
            return False
        if os.path.isdir(full) and not _is_dir_empty_recursive(full):
            return False
    return True

def _remove_empty_dirs(path, deleted):
    for entry in os.listdir(path):
        full = os.path.join(path, entry)
        if entry.startswith('.') or not os.path.isdir(full):
            continue
        _remove_empty_dirs(full, deleted)
        if _is_dir_empty_recursive(full):
            try:
                os.rmdir(full)
                deleted.append(os.path.relpath(full, folder_paths.models_dir))
            except:
                pass

# ===================== 健康检测 =====================
async def health_check(request):
    return web.json_response({
        "status": "ok",
        "service": "fxai-toolkit",
        "version": "1.0"
    })

# ===================== 文件夹管理 API =====================
async def get_folder(request):
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

async def list_files(request):
    dir_key = request.query.get("dir", "fxai")
    subdir = request.query.get("subdir", "")
    target = _resolve_target(dir_key, subdir)

    result = {"images": [], "audio": [], "video": [], "text": [], "other": []}
    if os.path.isdir(target):
        for name in sorted(os.listdir(target), key=str.lower):
            full = os.path.join(target, name)
            if os.path.isfile(full) and not name.startswith('.'):
                cat, item = _categorize_file(name, full)
                result[cat].append(item)

    return web.json_response(result)

async def list_subdirs(request):
    dir_key = request.query.get("dir", "fxai")
    subdir = request.query.get("subdir", "")
    target = _resolve_target(dir_key, subdir)

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

async def delete_file(request):
    dir_key = request.query.get("dir", "fxai")
    subdir = request.query.get("subdir", "")

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

    target = _resolve_target(dir_key, subdir)
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

async def delete_folder(request):
    dir_key = request.query.get("dir", "fxai")
    subdir = request.query.get("subdir", "")
    folder_name = request.query.get("folder", "")

    if not folder_name:
        return web.json_response({"error": "缺少 folder"}, status=400)
    folder_name = os.path.basename(folder_name)
    target = _resolve_target(dir_key, subdir)
    folder_path = os.path.join(target, folder_name)

    if not os.path.isdir(folder_path):
        return web.json_response({"error": "目录不存在"}, status=404)

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

async def create_folder(request):
    dir_key = request.query.get("dir", "fxai")
    subdir = request.query.get("subdir", "")
    folder_name = request.query.get("folder", "").strip()

    if not folder_name:
        return web.json_response({"error": "缺少 folder"}, status=400)
    folder_name = os.path.basename(folder_name)
    if not folder_name:
        return web.json_response({"error": "无效的目录名"}, status=400)
    target = _resolve_target(dir_key, subdir)
    folder_path = os.path.join(target, folder_name)

    if os.path.exists(folder_path):
        return web.json_response({"error": "目录已存在"}, status=409)

    try:
        os.makedirs(folder_path)
        return web.json_response({"success": True, "created": folder_name})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

# ===================== 代理 input/output 文件预览 =====================
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

# ===================== 文本预览 =====================
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

# ===================== 模型管理 API =====================
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

async def clean_empty_model_dirs(request):
    root = folder_paths.models_dir
    deleted = []
    if os.path.isdir(root):
        _remove_empty_dirs(root, deleted)
    return web.json_response({"success": True, "deleted": deleted})

# ===================== Workflows 目录 =====================
WORKFLOWS_DIR = os.path.join(os.path.dirname(__file__), "workflows")

async def list_workflows(request):
    files = []
    if os.path.isdir(WORKFLOWS_DIR):
        for name in sorted(os.listdir(WORKFLOWS_DIR), key=str.lower):
            if name.endswith(".json"):
                files.append({"filename": name, "title": name.replace(".json", "")})
    return web.json_response({"workflows": files})

WORKFLOWS_API_DIR = os.path.join(os.path.dirname(__file__), "workflows", "api")
if not os.path.isdir(WORKFLOWS_API_DIR):
    try: os.makedirs(WORKFLOWS_API_DIR)
    except: pass

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

# ===================== 系统控制 API =====================
async def shutdown_pc(request):
    try:
        data = await request.json()
        wait = int(data.get("wait_seconds", 60))
    except Exception:
        wait = 60
    wait = max(5, min(wait, 86400))
    import platform
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

async def reboot_pc(request):
    try:
        data = await request.json()
        wait = int(data.get("wait_seconds", 5))
    except Exception:
        wait = 5
    wait = max(0, min(wait, 86400))
    import platform
    sysos = platform.system()
    try:
        if sysos == "Windows":
            result = subprocess.run(["shutdown", "-r", "-t", str(wait)], capture_output=True, text=True)
            if result.returncode == 0:
                return web.json_response({"success": True, "message": f"已设置 {wait} 秒后重启"})
            return web.json_response({"success": False, "error": result.stderr.strip()}, status=500)
        elif sysos in ("Linux", "Darwin"):
            minutes = max(1, (wait + 59) // 60)
            result = subprocess.run(["sudo", "shutdown", "-r", f"+{minutes}"], capture_output=True, text=True)
            if result.returncode == 0:
                return web.json_response({"success": True, "message": f"已设置 {minutes} 分钟后重启"})
            return web.json_response({"success": False, "error": result.stderr.strip()}, status=500)
        else:
            return web.json_response({"success": False, "error": f"不支持的系统: {sysos}"}, status=400)
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def restart_comfyui(request):
    def _restart():
        import sys
        try:
            sys.stdout.close_log()
        except Exception:
            pass

        if '__COMFY_CLI_SESSION__' in os.environ:
            with open(os.path.join(os.environ['__COMFY_CLI_SESSION__'] + '.reboot'), 'w'):
                pass
            print("\nRestarting...\n\n", flush=True)
            exit(0)

        print("\nRestarting... [Legacy Mode]\n\n", flush=True)

        sys_argv = sys.argv.copy()
        if '--windows-standalone-build' in sys_argv:
            sys_argv.remove('--windows-standalone-build')

        if sys_argv[0].endswith("__main__.py"):
            module_name = os.path.basename(os.path.dirname(sys_argv[0]))
            cmds = [sys.executable, '-m', module_name] + sys_argv[1:]
        elif sys.platform.startswith('win32'):
            cmds = ['"' + sys.executable + '"', '"' + sys_argv[0] + '"'] + sys_argv[1:]
        else:
            cmds = [sys.executable] + sys_argv

        print(f"Command: {cmds}", flush=True)
        os.execv(sys.executable, cmds)

    try:
        await asyncio.to_thread(_restart)
        return web.json_response({"success": True, "message": "正在重启 ComfyUI 实例..."})
    except Exception as e:
        return web.json_response({"success": False, "error": f"重启失败: {e}"}, status=500)

# ===================== 图片管理 - V1 API =====================
def _image_v1_get_dir(subdir=""):
    comfy_root = folder_paths.base_path
    base_dir = "fxai/image"
    target_dir = os.path.join(comfy_root, base_dir)
    if subdir:
        subdir = re.sub(r'[\\/*?:"<>|]', "", subdir)
        target_dir = os.path.join(target_dir, subdir)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

def _image_v1_get_next_number(target_dir):
    used = set()
    if os.path.isdir(target_dir):
        for f in os.listdir(target_dir):
            m = re.match(r'^(\d+)', f)
            if m:
                used.add(int(m.group(1)))
    next_num = 0
    while next_num in used:
        next_num += 1
    return next_num

def _image_v1_list_images(target_dir):
    if not os.path.isdir(target_dir):
        return []
    pattern = re.compile(r'(.+)\.(png|jpg|jpeg|webp)$', re.IGNORECASE)
    files = []
    for f in os.listdir(target_dir):
        fp = os.path.join(target_dir, f)
        if not os.path.isfile(fp):
            continue
        m = pattern.match(f)
        if m:
            files.append((m.group(1), f))
    files.sort()
    return [f for _, f in files]

async def image_v1_preview(request):
    subdir = request.query.get("subdir", "")
    filename = request.query.get("filename", "")
    if not filename:
        return web.json_response({"error": "未提供文件名"}, status=400)

    target_dir = _image_v1_get_dir(subdir)
    safe_file = _safe_path_join(target_dir, filename)
    if not safe_file or not os.path.exists(safe_file):
        return web.json_response({"error": "文件未找到"}, status=404)

    return web.FileResponse(safe_file, headers={
        "Content-Type": mimetypes.guess_type(safe_file)[0] or "image/png",
        "Cache-Control": "no-store, no-cache, must-revalidate"
    })

async def image_v1_next_number(request):
    subdir = request.query.get("subdir", "")
    target_dir = _image_v1_get_dir(subdir)
    next_num = _image_v1_get_next_number(target_dir)
    return web.json_response({"next_num": next_num})

async def image_v1_list(request):
    subdir = request.query.get("subdir", "")
    target_dir = _image_v1_get_dir(subdir)
    files = _image_v1_list_images(target_dir)
    return web.json_response({"files": files, "total": len(files)})

async def image_v1_apply(request):
    try:
        data = await request.json()
        subdir = data.get("subdir", "")
        ordered_filenames = data.get("ordered_filenames", [])
        target_dir = _image_v1_get_dir(subdir)

        existing_files = _image_v1_list_images(target_dir)
        existing_set = set(existing_files)
        safe_ordered = [f for f in ordered_filenames if f in existing_set]

        to_delete = existing_set - set(safe_ordered)
        for f in to_delete:
            fp = _safe_path_join(target_dir, f)
            if fp:
                os.remove(fp)

        temp_map = []
        for idx, old_name in enumerate(safe_ordered):
            old_fp = _safe_path_join(target_dir, old_name)
            if not old_fp or not os.path.exists(old_fp):
                continue

            ext = old_name.split('.')[-1].lower()
            new_name = f"{idx:03d}.{ext}"
            temp_name = f"_tmp_{idx}_{os.urandom(4).hex()}_{old_name}"
            temp_fp = _safe_path_join(target_dir, temp_name)

            os.rename(old_fp, temp_fp)
            temp_map.append((temp_fp, new_name))

        for temp_fp, new_name in temp_map:
            final_fp = _safe_path_join(target_dir, new_name)
            if temp_fp and final_fp:
                os.rename(temp_fp, final_fp)

        new_files = _image_v1_list_images(target_dir)
        return web.json_response({"files": new_files, "success": True})
    except Exception as e:
        return web.json_response({"error": f"应用失败：{str(e)}"}, status=500)

async def image_v1_upload(request):
    try:
        data = await request.post()
        images = data.getall("image")
        subdir = data.get("subdir", "")

        if not images:
            return web.json_response({"error": "未上传有效图片"}, status=400)

        target_dir = _image_v1_get_dir(subdir)
        results = []
        for image in images:
            if not image or not hasattr(image, 'file'):
                continue

            original_filename = re.sub(r'[\\/*?:"<>|]', "", image.filename)
            if not original_filename:
                continue

            next_num = _image_v1_get_next_number(target_dir)

            ext = original_filename.split('.')[-1].lower()
            if ext not in ['png', 'jpg', 'jpeg', 'webp']:
                ext = 'png'

            new_filename = f"{next_num:03d}.{ext}"
            save_path = _safe_path_join(target_dir, new_filename)

            with open(save_path, "wb") as f:
                f.write(image.file.read())

            results.append(new_filename)

        if not results:
            return web.json_response({"error": "未上传有效图片"}, status=400)

        if len(results) == 1:
            return web.json_response({"success": True, "name": results[0]})
        return web.json_response({"success": True, "names": results, "count": len(results)})
    except Exception as e:
        return web.json_response({"error": f"上传失败：{str(e)}"}, status=500)

# ===================== 图片管理 - V2 API =====================
def _image_v2_get_next_number(target_dir):
    files = _image_v1_list_images(target_dir)
    used = set()
    for f in files:
        m = re.match(r'^(\d+)', f)
        if m:
            used.add(int(m.group(1)))
    n = 0
    while n in used:
        n += 1
    return n

async def image_v2_preview(request):
    subdir = request.query.get("subdir", "")
    filename = request.query.get("filename", "")
    if not filename:
        return web.json_response({"error": "未提供文件名"}, status=400)

    target_dir = _image_v1_get_dir(subdir)
    safe_file = _safe_path_join(target_dir, filename)
    if not safe_file or not os.path.exists(safe_file):
        return web.json_response({"error": "文件未找到"}, status=404)

    return web.FileResponse(safe_file, headers={
        "Content-Type": mimetypes.guess_type(safe_file)[0] or "image/png",
        "Cache-Control": "no-store, no-cache, must-revalidate"
    })

async def image_v2_list(request):
    subdir = request.query.get("subdir", "")
    target_dir = _image_v1_get_dir(subdir)
    files = _image_v1_list_images(target_dir)
    return web.json_response({"files": files, "total": len(files)})

async def image_v2_apply(request):
    try:
        data = await request.json()
        subdir = data.get("subdir", "")
        ordered_filenames = data.get("ordered_filenames", [])
        target_dir = _image_v1_get_dir(subdir)

        if not ordered_filenames:
            return web.json_response({"files": _image_v1_list_images(target_dir), "success": True, "msg": "无需修改"})

        for idx, old_name in enumerate(ordered_filenames):
            old_path = _safe_path_join(target_dir, old_name)
            if not old_path or not os.path.exists(old_path):
                continue
            temp_name = f"{idx:03d}_{old_name}"
            temp_path = _safe_path_join(target_dir, temp_name)
            os.rename(old_path, temp_path)

        for idx, old_name in enumerate(ordered_filenames):
            temp_name = f"{idx:03d}_{old_name}"
            temp_path = _safe_path_join(target_dir, temp_name)
            if not temp_path or not os.path.exists(temp_path):
                continue

            ext = old_name.split(".")[-1]
            final_name = f"{idx:03d}.{ext}"
            final_path = _safe_path_join(target_dir, final_name)
            os.rename(temp_path, final_path)

        new_files = _image_v1_list_images(target_dir)
        return web.json_response({"files": new_files, "success": True})
    except Exception as e:
        return web.json_response({"error": f"应用失败：{str(e)}"}, status=500)

async def image_v2_upload(request):
    try:
        data = await request.post()
        images = data.getall("image")
        subdir = data.get("subdir", "")

        if not images:
            return web.json_response({"error": "未上传有效图片"}, status=400)

        target_dir = _image_v1_get_dir(subdir)
        results = []
        for image in images:
            if not image or not hasattr(image, 'file'):
                continue

            original_filename = re.sub(r'[\\/*?:"<>|]', "", image.filename)
            if not original_filename:
                continue

            next_num = _image_v2_get_next_number(target_dir)

            ext = original_filename.split('.')[-1].lower()
            if ext not in ['png', 'jpg', 'jpeg', 'webp']:
                ext = 'png'

            new_filename = f"{next_num:03d}.{ext}"
            save_path = _safe_path_join(target_dir, new_filename)

            with open(save_path, "wb") as f:
                f.write(image.file.read())

            results.append(new_filename)

        if not results:
            return web.json_response({"error": "未上传有效图片"}, status=400)

        if len(results) == 1:
            return web.json_response({"success": True, "name": results[0]})
        return web.json_response({"success": True, "names": results, "count": len(results)})
    except Exception as e:
        return web.json_response({"error": f"上传失败：{str(e)}"}, status=500)

async def image_v2_delete(request):
    try:
        subdir = request.query.get("subdir", "")
        filename = request.query.get("filename", "")
        if not filename:
            return web.json_response({"error": "未提供文件名"}, status=400)

        target_dir = _image_v1_get_dir(subdir)
        safe_file = _safe_path_join(target_dir, filename)

        if not safe_file or not os.path.exists(safe_file):
            return web.json_response({"error": "文件不存在"}, status=404)

        os.remove(safe_file)
        return web.json_response({"success": True})
    except Exception as e:
        return web.json_response({"error": f"删除失败：{str(e)}"}, status=500)

# ===================== 音频管理 API =====================
def _audio_get_dir(subdir=""):
    comfy_root = folder_paths.base_path
    base_dir = "fxai/audio"
    target_dir = os.path.join(comfy_root, base_dir)
    if subdir:
        subdir = re.sub(r'[\\/*?:"<>|]', "", subdir)
        target_dir = os.path.join(target_dir, subdir)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

def _audio_list_audios(target_dir):
    if not os.path.isdir(target_dir):
        return []
    pattern = re.compile(r'(.+)\.(mp3|wav|ogg|flac|m4a)$', re.IGNORECASE)
    files = []
    for f in os.listdir(target_dir):
        fp = os.path.join(target_dir, f)
        if not os.path.isfile(fp):
            continue
        m = pattern.match(f)
        if m:
            files.append(f)
    files.sort()
    return files

def _audio_get_duration_ffprobe(audio_path):
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-print_format", "json", "-show_format",
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=10)
        meta = json.loads(result.stdout)
        duration = float(meta["format"]["duration"])
        return round(duration, 3)
    except Exception:
        return 0.0

async def audio_preview(request):
    subdir = request.query.get("subdir", "")
    filename = request.query.get("filename", "")
    if not filename:
        return web.json_response({"error": "未提供文件名"}, status=400)

    target_dir = _audio_get_dir(subdir)
    safe_file = _safe_path_join(target_dir, filename)
    if not safe_file or not os.path.exists(safe_file):
        return web.json_response({"error": "文件未找到"}, status=404)

    mime_type = mimetypes.guess_type(safe_file)[0]
    if not mime_type:
        ext = filename.split('.')[-1].lower()
        mime_map = {
            'mp3': 'audio/mpeg', 'wav': 'audio/wav', 'ogg': 'audio/ogg',
            'flac': 'audio/flac', 'm4a': 'audio/mp4'
        }
        mime_type = mime_map.get(ext, "audio/mpeg")

    return web.FileResponse(safe_file, headers={
        "Content-Type": mime_type,
        "Cache-Control": "no-store, no-cache, must-revalidate"
    })

async def audio_list(request):
    subdir = request.query.get("subdir", "")
    target_dir = _audio_get_dir(subdir)
    files = _audio_list_audios(target_dir)
    return web.json_response({"files": files, "total": len(files)})

async def audio_apply(request):
    try:
        data = await request.json()
        subdir = data.get("subdir", "")
        ordered_filenames = data.get("ordered_filenames", [])
        target_dir = _audio_get_dir(subdir)

        existing_files = _audio_list_audios(target_dir)
        existing_set = set(existing_files)
        safe_ordered = [f for f in ordered_filenames if f in existing_set]

        to_delete = existing_set - set(safe_ordered)
        for f in to_delete:
            fp = _safe_path_join(target_dir, f)
            if fp:
                os.remove(fp)

        temp_map = []
        for idx, old_fullname in enumerate(safe_ordered):
            old_fp = _safe_path_join(target_dir, old_fullname)
            if not old_fp or not os.path.exists(old_fp):
                continue

            match = re.match(r'^\d{3}_(.+)', os.path.splitext(old_fullname)[0])
            if match:
                pure_name = match.group(1)
            else:
                pure_name = os.path.splitext(old_fullname)[0]

            ext = old_fullname.split('.')[-1].lower()
            new_name = f"{idx:03d}_{pure_name}.{ext}"

            temp_name = f"_tmp_{os.urandom(4).hex()}"
            temp_fp = _safe_path_join(target_dir, temp_name)
            os.rename(old_fp, temp_fp)
            temp_map.append((temp_fp, new_name))

        for temp_fp, new_name in temp_map:
            final_fp = _safe_path_join(target_dir, new_name)
            os.rename(temp_fp, final_fp)

        new_files = _audio_list_audios(target_dir)
        return web.json_response({"files": new_files, "success": True})
    except Exception as e:
        return web.json_response({"error": f"应用失败：{str(e)}"}, status=500)

async def audio_upload(request):
    try:
        data = await request.post()
        audios = data.getall("audio")
        subdir = data.get("subdir", "")

        if not audios:
            return web.json_response({"error": "未上传有效音频"}, status=400)

        target_dir = _audio_get_dir(subdir)
        results = []
        for audio in audios:
            if not audio or not hasattr(audio, 'file'):
                continue

            original_filename = _sanitize_filename(audio.filename)
            if not original_filename:
                continue

            file_list = _audio_list_audios(target_dir)
            next_num = len(file_list)

            base_name = os.path.splitext(original_filename)[0]
            raw_path = _safe_path_join(target_dir, original_filename)
            with open(raw_path, "wb") as f:
                f.write(audio.file.read())

            new_filename = f"{next_num:03d}_{base_name}.wav"
            wav_path = _safe_path_join(target_dir, new_filename)
            cmd = [
                "ffmpeg", "-i", raw_path,
                "-f", "wav", "-y", wav_path
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)

            if os.path.exists(wav_path):
                try:
                    os.remove(raw_path)
                except Exception:
                    pass
            else:
                new_filename = original_filename

            results.append(new_filename)

        if not results:
            return web.json_response({"error": "未上传有效音频"}, status=400)

        if len(results) == 1:
            return web.json_response({"success": True, "name": results[0]})
        return web.json_response({"success": True, "names": results, "count": len(results)})
    except Exception as e:
        return web.json_response({"error": f"上传失败：{str(e)}"}, status=500)

async def audio_delete(request):
    try:
        subdir = request.query.get("subdir", "")
        filename = request.query.get("filename", "")

        if not filename:
            return web.json_response({"error": "未提供文件名"}, status=400)

        target_dir = _audio_get_dir(subdir)
        safe_file = _safe_path_join(target_dir, filename)

        if not safe_file or not os.path.exists(safe_file):
            return web.json_response({"error": "文件未找到"}, status=404)

        os.remove(safe_file)

        return web.json_response({"success": True})
    except Exception as e:
        return web.json_response({"error": f"删除失败：{str(e)}"}, status=500)

# ===================== 视频管理 API =====================
def _video_get_dir(subdir=""):
    comfy_root = folder_paths.base_path
    base_dir = "fxai/video"
    target_dir = os.path.join(comfy_root, base_dir)
    if subdir:
        subdir = re.sub(r'[\\/*?:"<>|]', "", subdir)
        target_dir = os.path.join(target_dir, subdir)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

def _video_list_videos(target_dir):
    if not os.path.isdir(target_dir):
        return []
    pattern = re.compile(r'(.+)\.(mp4|mov|avi|mkv|flv|wmv|webm)$', re.IGNORECASE)
    files = []
    for f in os.listdir(target_dir):
        fp = os.path.join(target_dir, f)
        if not os.path.isfile(fp):
            continue
        m = pattern.match(f)
        if m:
            files.append(f)
    return files

async def video_loop_preview(request):
    subdir = request.query.get("subdir", "")
    filename = request.query.get("filename", "")
    if not filename:
        return web.json_response({"error": "未提供文件名"}, status=400)

    target_dir = _video_get_dir(subdir)
    safe_file = _safe_path_join(target_dir, filename)
    if not safe_file or not os.path.exists(safe_file):
        return web.json_response({"error": "文件未找到"}, status=404)

    mime_type = mimetypes.guess_type(safe_file)[0]
    if not mime_type:
        ext = filename.split('.')[-1].lower()
        mime_map = {
            'mp4': 'video/mp4', 'mov': 'video/quicktime', 'avi': 'video/x-msvideo',
            'mkv': 'video/x-matroska', 'flv': 'video/x-flv', 'wmv': 'video/x-ms-wmv',
            'webm': 'video/webm'
        }
        mime_type = mime_map.get(ext, "video/mp4")

    return web.FileResponse(safe_file, headers={
        "Content-Type": mime_type,
        "Cache-Control": "no-store, no-cache, must-revalidate"
    })

async def video_list(request):
    subdir = request.query.get("subdir", "")
    target_dir = _video_get_dir(subdir)
    files = _video_list_videos(target_dir)
    return web.json_response({"files": files, "total": len(files)})

async def video_apply(request):
    try:
        data = await request.json()
        subdir = data.get("subdir", "")
        ordered_filenames = data.get("ordered_filenames", [])
        target_dir = _video_get_dir(subdir)

        existing_files = _video_list_videos(target_dir)
        existing_set = set(existing_files)
        safe_ordered = [f for f in ordered_filenames if f in existing_set]

        to_delete = existing_set - set(safe_ordered)
        for f in to_delete:
            fp = _safe_path_join(target_dir, f)
            if fp and os.path.exists(fp):
                os.remove(fp)

        temp_map = []
        for idx, old_fullname in enumerate(safe_ordered):
            old_fp = _safe_path_join(target_dir, old_fullname)
            if not old_fp or not os.path.exists(old_fp):
                continue

            match = re.match(r'^\d{3}_(.+)', os.path.splitext(old_fullname)[0])
            if match:
                pure_name = match.group(1)
            else:
                pure_name = os.path.splitext(old_fullname)[0]

            ext = old_fullname.split('.')[-1].lower()
            new_name = f"{idx:03d}_{pure_name}.{ext}"

            temp_name = f"_tmp_{os.urandom(4).hex()}"
            temp_fp = _safe_path_join(target_dir, temp_name)
            os.rename(old_fp, temp_fp)
            temp_map.append((temp_fp, new_name))

        for temp_fp, new_name in temp_map:
            final_fp = _safe_path_join(target_dir, new_name)
            os.rename(temp_fp, final_fp)

        new_files = _video_list_videos(target_dir)
        return web.json_response({"files": new_files, "success": True})
    except Exception as e:
        return web.json_response({"error": f"应用失败：{str(e)}"}, status=500)

async def video_upload(request):
    try:
        data = await request.post()
        videos = data.getall("video")
        subdir = data.get("subdir", "")

        if not videos:
            return web.json_response({"error": "未上传有效视频"}, status=400)

        target_dir = _video_get_dir(subdir)
        results = []
        for video in videos:
            if not video or not hasattr(video, 'file'):
                continue

            original_filename = _sanitize_filename(video.filename)
            if not original_filename:
                continue

            file_list = _video_list_videos(target_dir)
            next_num = len(file_list)

            new_filename = f"{next_num:03d}_{original_filename}"

            save_path = _safe_path_join(target_dir, new_filename)
            with open(save_path, "wb") as f:
                f.write(video.file.read())

            results.append(new_filename)

        if not results:
            return web.json_response({"error": "未上传有效视频"}, status=400)

        if len(results) == 1:
            return web.json_response({"success": True, "name": results[0]})
        return web.json_response({"success": True, "names": results, "count": len(results)})
    except Exception as e:
        return web.json_response({"error": f"上传失败：{str(e)}"}, status=500)

async def video_delete(request):
    try:
        subdir = request.query.get("subdir", "")
        filename = request.query.get("filename", "")

        if not filename:
            return web.json_response({"error": "未提供文件名"}, status=400)

        target_dir = _video_get_dir(subdir)
        safe_file = _safe_path_join(target_dir, filename)

        if not safe_file or not os.path.exists(safe_file):
            return web.json_response({"error": "文件未找到"}, status=404)

        os.remove(safe_file)

        return web.json_response({"success": True})
    except Exception as e:
        return web.json_response({"error": f"删除失败：{str(e)}"}, status=500)

# ===================== 视频预览（带时间戳文件名） =====================
async def video_preview(request):
    path = request.query.get("path")
    if not path or not os.path.exists(path):
        return web.Response(status=404)

    time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"fxai_{time_str}.mp4"

    headers = {
        "Content-Type": "video/mp4",
        "Content-Disposition": f'inline; filename="{filename}"'
    }

    return web.FileResponse(path, headers=headers)

# ===================== 任务查询 API =====================
async def task_query(request):
    from fxai_task_store import get_tasks
    if request.method == "POST":
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "需要 JSON body"}, status=400)
    else:
        data = request.query
    prompt_ids = data.get("prompt_ids", "") or data.get("prompt_id", "")
    if not prompt_ids:
        return web.json_response({"error": "缺少 prompt_ids"}, status=400)
    return web.json_response(get_tasks(prompt_ids))

# ===================== 提示词管理 API =====================
def _prompt_get_dir(subdir=""):
    comfy_root = folder_paths.base_path
    base_dir = "fxai/prompts"
    target_dir = os.path.join(comfy_root, base_dir)
    if subdir:
        subdir = re.sub(r'[\\/*?:"<>|]', "", subdir)
        subdir = os.path.normpath(subdir).lstrip(".")
        target_dir = os.path.join(target_dir, subdir)
    safe_dir = _safe_path_join(comfy_root, os.path.relpath(target_dir, comfy_root))
    if not safe_dir:
        return os.path.join(comfy_root, base_dir)
    os.makedirs(safe_dir, exist_ok=True)
    return safe_dir

def _prompt_list_prompts(target_dir):
    if not os.path.isdir(target_dir):
        return []
    files = []
    for f in os.listdir(target_dir):
        fp = os.path.join(target_dir, f)
        if os.path.isfile(fp) and f.lower().endswith(".txt"):
            files.append(f)
    files.sort()
    return files

async def prompt_list(request):
    subdir = request.query.get("subdir", "")
    target_dir = _prompt_get_dir(subdir)
    files = _prompt_list_prompts(target_dir)
    return web.json_response({"files": files, "total": len(files)})

async def prompt_save(request):
    try:
        data = await request.post()
        subdir = data.get("subdir", "")
        filename = data.get("filename", "").strip()
        content = data.get("content", "").strip()

        if not filename:
            return web.json_response({"error": "文件名不能为空"}, status=400)
        if not content:
            return web.json_response({"error": "提示词内容不能为空"}, status=400)

        filename = re.sub(r'[\\/*?:"<>|]', "", filename)
        if not filename.lower().endswith(".txt"):
            filename += ".txt"

        target_dir = _prompt_get_dir(subdir)
        save_path = _safe_path_join(target_dir, filename)
        if not save_path:
            return web.json_response({"error": "非法路径"}, status=403)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)

        return web.json_response({"success": True, "name": filename})
    except Exception as e:
        return web.json_response({"error": f"保存失败：{str(e)}"}, status=500)

async def prompt_delete(request):
    subdir = request.query.get("subdir", "")
    filename = request.query.get("filename", "")
    if not filename:
        return web.json_response({"error": "未提供文件名"}, status=400)

    target_dir = _prompt_get_dir(subdir)
    safe_file = _safe_path_join(target_dir, filename)
    if not safe_file or not os.path.exists(safe_file):
        return web.json_response({"error": "文件不存在"}, status=404)

    try:
        os.remove(safe_file)
        return web.json_response({"success": True, "name": filename})
    except Exception as e:
        return web.json_response({"error": f"删除失败：{str(e)}"}, status=500)

# ===================== 提示词优化 - Ollama 模型 API =====================
async def prompt_get_models(request):
    host = request.query.get("host", "http://127.0.0.1:11434")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{host}/api/tags", timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m["name"] for m in data.get("models", [])]
                    return web.json_response({"models": models if models else [""]})
    except Exception:
        pass
    return web.json_response({"models": [""]})

# ===================== LoRA 管理 API =====================
def _lora_safe_float(val, default=1.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def _lora_extract_trigger_words(lora_path):
    try:
        import torch
        import safetensors
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

def _lora_get_config_dir():
    root = folder_paths.base_path
    cfg_dir = os.path.join(root, "fxai", "loras")
    os.makedirs(cfg_dir, exist_ok=True)
    return cfg_dir

def _lora_clean_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", str(name).strip())

def _lora_get_config_path(lora_name):
    cfg_dir = _lora_get_config_dir()
    pure_name = os.path.splitext(_lora_clean_filename(lora_name))[0]
    return _safe_path_join(cfg_dir, f"{pure_name}.json")

def _lora_load_config(lora_name):
    path = _lora_get_config_path(lora_name)
    default_config = {
        "enabled": True,
        "model_strength": 1.0,
        "clip_strength": -1.0,
        "trigger_words": [],
        "invert": False,
        "fade_start": 1.0,
        "fade_end": 1.0
    }

    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
                if "model_strength" in user_config:
                    user_config["model_strength"] = _lora_safe_float(user_config["model_strength"], 1.0)
                if "clip_strength" in user_config:
                    user_config["clip_strength"] = _lora_safe_float(user_config["clip_strength"], -1.0)
                if "fade_start" in user_config:
                    user_config["fade_start"] = _lora_safe_float(user_config["fade_start"], 1.0)
                if "fade_end" in user_config:
                    user_config["fade_end"] = _lora_safe_float(user_config["fade_end"], 1.0)
                default_config.update(user_config)
        except Exception:
            pass

    lora_path = folder_paths.get_full_path("loras", lora_name)
    if lora_path:
        triggers = _lora_extract_trigger_words(lora_path)
        default_config["trigger_words"] = triggers

    if path:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return default_config

def _lora_load_config_advanced(lora_name):
    path = _lora_get_config_path(lora_name)
    default = {
        "enable": True,
        "model_strength": 1.0,
        "clip_strength": 1.0,
        "trigger_words": [],
        "invert": False,
        "fade_start": 1.0,
        "fade_end": 1.0,
        "layer_mode": "all"
    }
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {**default, **json.load(f)}
    except Exception:
        return default

async def lora_files(request):
    loras = folder_paths.get_filename_list("loras")
    loras_sorted = sorted(loras, key=lambda x: x.lower())
    result = {}
    for lora_name in loras_sorted:
        result[lora_name] = _lora_load_config(lora_name)
    return web.json_response(result)

async def lora_config(request):
    name = request.query.get("name", "")
    return web.json_response(_lora_load_config_advanced(name))

async def lora_list(request):
    files = []
    cfg_dir = _lora_get_config_dir()
    for f in os.listdir(cfg_dir):
        if f.lower().endswith(".json"):
            files.append(os.path.splitext(f)[0])
    return web.json_response(sorted(files))

# ===================== 节点更新 API =====================
NODE_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_NAME = "origin"
REMOTE_BRANCH = "main"
GIT_TIMEOUT = 300

def _git(args):
    cmd = ["git", "-C", NODE_DIR] + args
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=GIT_TIMEOUT,
        )
        output = (proc.stdout or "").strip()
        error = (proc.stderr or "").strip()
        merged = output if output else error
        return proc.returncode, merged
    except FileNotFoundError:
        return -1, "未找到 git 命令，请确认服务器已安装 git 并加入 PATH"
    except subprocess.TimeoutExpired:
        return -1, "git 命令执行超时"

async def node_status(request):
    local_commit = ""
    local_branch = ""
    remote_commit = ""
    remote_branch = ""
    remote_url = ""
    remote_reachable = False
    message = ""

    code, out = await asyncio.to_thread(_git, ["rev-parse", "--short", "HEAD"])
    if code != 0:
        return web.json_response({
            "status": "error",
            "message": f"获取本地版本失败：{out}",
            "node_dir": NODE_DIR,
        })
    local_commit = out

    code, out = await asyncio.to_thread(_git, ["rev-parse", "--abbrev-ref", "HEAD"])
    if code == 0:
        local_branch = out

    code, out = await asyncio.to_thread(_git, ["remote", "get-url", REMOTE_NAME])
    if code == 0:
        remote_url = out

    code, out = await asyncio.to_thread(_git, ["ls-remote", REMOTE_NAME, f"refs/heads/{REMOTE_BRANCH}"])
    if code == 0:
        remote_reachable = True
        parts = out.split()
        if parts:
            remote_commit = parts[0][:7]
            remote_branch = REMOTE_BRANCH
    else:
        message = f"无法连接远程仓库（ls-remote 失败）：{out}"

    return web.json_response({
        "status": "ok",
        "message": message,
        "node_dir": NODE_DIR,
        "local_commit": local_commit,
        "local_branch": local_branch,
        "remote_commit": remote_commit,
        "remote_branch": remote_branch,
        "remote_url": remote_url,
        "remote_reachable": remote_reachable,
        "has_update": bool(local_commit and remote_commit and local_commit != remote_commit),
        "git_available": True,
    })

async def node_update(request):
    logs = []

    logs.append(f"[1/2] git fetch {REMOTE_NAME} ...")
    code, out = await asyncio.to_thread(_git, ["fetch", REMOTE_NAME, "--prune"])
    logs.append(out if out else ("成功" if code == 0 else "无输出"))
    if code != 0:
        logs.append("更新失败：fetch 出错，已终止（本地代码未被修改）")
        return web.json_response({"status": "error", "message": f"git fetch 失败：{out}", "logs": logs})

    logs.append(f"[2/2] git reset --hard {REMOTE_NAME}/{REMOTE_BRANCH} ...")
    code, out = await asyncio.to_thread(_git, ["reset", "--hard", f"{REMOTE_NAME}/{REMOTE_BRANCH}"])
    logs.append(out if out else ("成功" if code == 0 else "无输出"))
    if code != 0:
        logs.append("更新失败：reset --hard 出错")
        return web.json_response({"status": "error", "message": f"git reset 失败：{out}", "logs": logs})

    new_commit = ""
    code, out = await asyncio.to_thread(_git, ["rev-parse", "--short", "HEAD"])
    if code == 0:
        new_commit = out

    logs.append(f"更新完成，当前版本：{new_commit}")
    return web.json_response({"status": "ok", "message": "更新完成", "logs": logs, "new_commit": new_commit})

# ===================== 音频分段 API（V1 - 旧版，保留兼容） =====================
MAX_MARKERS = 64

def _seg_strip_path(path):
    path = (path or "").strip()
    if path.startswith('"'):
        path = path[1:]
    if path.endswith('"'):
        path = path[:-1]
    return path

def _seg_resolve_audio_path(audio_file):
    audio_file = _seg_strip_path(audio_file)
    if not audio_file:
        raise ValueError("音频文件路径为空")
    if os.path.isabs(audio_file) and os.path.isfile(audio_file):
        return audio_file
    try:
        annotated = folder_paths.get_annotated_filepath(audio_file)
        if annotated and os.path.isfile(annotated):
            return annotated
    except Exception:
        pass
    input_candidate = os.path.join(folder_paths.get_input_directory(), audio_file)
    if os.path.isfile(input_candidate):
        return input_candidate
    audio_root = os.path.join(folder_paths.base_path, "fxai", "audio")
    path_candidate = os.path.join(audio_root, audio_file)
    if os.path.isfile(path_candidate):
        return path_candidate
    raise ValueError(f"未找到音频文件: {audio_file}")

def _seg_load_audio_tensor_from_file(audio_file):
    audio_path = _seg_resolve_audio_path(audio_file)
    ext = os.path.splitext(audio_path)[1].lower()

    if ext != ".wav":
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(audio_file)
            sr = seg.frame_rate
            seg = seg.set_channels(2)
            raw = np.array(seg.get_array_of_samples(), dtype=np.float32)
            max_val = 1 << (8 * seg.sample_width - 1)
            if max_val > 0:
                raw /= max_val
            raw = raw.reshape(-1, 2).T
            waveform = torch.from_numpy(raw).unsqueeze(0).float()
            return {"waveform": waveform, "sample_rate": sr}
        except ImportError:
            raise ValueError("未安装pydub，请执行: pip install pydub")
        except Exception as e:
            raise ValueError(f"pydub处理失败: {e}\n请确保ffmpeg已添加到系统PATH")

    import soundfile as sf
    arr, sr = sf.read(audio_path)
    arr = arr.astype(np.float32)
    if arr.ndim == 1:
        arr = np.stack([arr, arr], axis=0)
    elif arr.shape[1] == 1:
        arr = np.repeat(arr, 2, axis=1).T
    else:
        arr = arr[:, :2].T
    waveform = torch.from_numpy(arr).unsqueeze(0).float()
    return {"waveform": waveform, "sample_rate": sr}

def _seg_read_waveform_peaks(audio_file, bins=1400):
    audio = _seg_load_audio_tensor_from_file(audio_file)
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    waveform_np = waveform.numpy()
    if waveform_np.shape[1] > 1:
        samples = np.mean(np.abs(waveform_np), axis=1)[0]
    else:
        samples = np.abs(waveform_np[0, 0])
    frame_count = len(samples)
    bins = max(64, min(int(bins), 4096))
    if samples.size == 0:
        peaks = []
    else:
        edges = np.linspace(0, samples.size, num=bins + 1, dtype=np.int64)
        peaks = []
        for idx in range(bins):
            start = edges[idx]
            end = edges[idx + 1]
            if end <= start:
                peaks.append(0.0)
                continue
            peaks.append(float(np.max(samples[start:end])))
    duration = float(frame_count) / float(sample_rate) if sample_rate > 0 else 0.0
    return {
        "duration": duration,
        "sample_rate": sample_rate,
        "peaks": peaks,
        "audio_path": _seg_resolve_audio_path(audio_file),
    }

def _seg_parse_keyframe_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        text = str(value).strip()
        if not text:
            return []
        parsed = json.loads(text)
        raw = parsed.get("keyframes", []) if isinstance(parsed, dict) else parsed
    return [max(0.0, float(x)) for x in raw]

def _seg_normalize_keyframe_list(keyframes, total_duration=None):
    seen = set()
    norm = []
    for sec in keyframes or []:
        sec = max(0.0, float(sec))
        if total_duration is not None and total_duration > 0:
            sec = min(sec, total_duration - 0.001)
        bucket = int(round(sec * 1000))
        if bucket not in seen:
            seen.add(bucket)
            norm.append(sec)
    norm.sort()
    return norm[:MAX_MARKERS]

def _seg_normalize_audio_tensor(audio):
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    return waveform, sample_rate

def _seg_slice_audio(audio, start_frame, end_frame):
    waveform, sample_rate = _seg_normalize_audio_tensor(audio)
    start_frame = max(0, int(start_frame))
    end_frame = max(start_frame + 1, int(end_frame))
    return {
        "waveform": waveform[..., start_frame:end_frame],
        "sample_rate": sample_rate,
    }

def _seg_build_segments(total_duration, keyframes, skip_initial_segment, include_tail_segment, is_average_split=False, average_duration=0.0):
    total_duration = max(0.0, total_duration)
    segments = []
    markers = _seg_normalize_keyframe_list(keyframes, total_duration)
    if not markers:
        segments = [(0.0, total_duration)]
    else:
        points = [0.0] + markers + [total_duration]
        for i in range(len(points) - 1):
            s = points[i]
            e = points[i + 1]
            if e > s:
                segments.append((s, e))

    if skip_initial_segment and len(segments) > 0:
        segments = segments[1:]
    if not include_tail_segment and len(segments) > 0:
        segments = segments[:-1]

    if len(segments) > 0:
        last_s, last_e = segments[-1]
        if (last_e - last_s) < 0.1:
            segments.pop()

    if not segments:
        segments = [(0.0, total_duration)]

    if is_average_split and average_duration > 0:
        if len(segments) == 0:
            start_total = 0.0
            end_total = total_duration
        else:
            start_total = segments[0][0]
            end_total = segments[-1][1]

        new_segments = []
        current = start_total
        while current < end_total:
            end = current + average_duration
            if end > end_total:
                end = end_total
            new_segments.append((current, end))
            current = end

        segments = new_segments

    if not segments:
        segments = [(0.0, total_duration)]

    total_selected = sum(e - s for s, e in segments)
    return segments, total_selected

def _seg_safe_int(value, default=0):
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return default
        return int(value)
    except Exception:
        return default

def _seg_list_input_audio_files():
    input_dir = folder_paths.get_input_directory()
    if not input_dir or not os.path.isdir(input_dir):
        return []
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    discovered = []
    for root, _dirs, files in os.walk(input_dir):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in audio_extensions:
                continue
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, input_dir).replace("\\", "/")
            discovered.append(rel_path)
    return sorted(discovered)

async def audio_segments_file(request):
    import torch
    import numpy as np
    audio_file = request.query.get("audio_file", "")
    try:
        path = _seg_resolve_audio_path(audio_file)
        return web.FileResponse(path, headers={"Content-Type": mimetypes.guess_type(path)[0] or "application/octet-stream"})
    except Exception as e:
        return web.JsonResponse({"error": str(e)}, status=400)

async def audio_segments_waveform(request):
    import torch
    import numpy as np
    audio_file = request.query.get("audio_file", "")
    bins = request.query.get("bins", "1400")
    try:
        data = _seg_read_waveform_peaks(audio_file, bins=int(bins))
        data["audio_url"] = f"/fxai/audio-file?audio_file={audio_file}"
        return web.json_response(data)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)

# ===================== 音频分段 API（V2） =====================
from fxai_audio_utils import (
    resolve_audio_path,
    load_audio_tensor_from_file,
    normalize_audio_tensor,
    slice_audio,
    read_waveform_peaks,
    get_wav_path,
    AUDIO_EXTENSIONS
)

async def audio_segments_v2_file(request):
    audio_file = request.query.get("audio_file", "")
    try:
        path = resolve_audio_path(audio_file)
        return web.FileResponse(path, headers={"Content-Type": "audio/wav"})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)

async def audio_segments_v2_waveform(request):
    audio_file = request.query.get("audio_file", "")
    bins = request.query.get("bins", "1400")
    try:
        data = read_waveform_peaks(audio_file, bins=int(bins))
        wav_file = get_wav_path(audio_file)
        data["audio_url"] = f"/fxai/audio/segments/file?audio_file={wav_file}"
        return web.json_response(data)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)

# ===================== 角色管理 API =====================
from fxai_character_profile_manager import (
    list_characters as _char_list,
    save_character as _char_save,
    delete_character as _char_delete,
    get_characters_by_avatars as _char_by_avatars,
    save_characters_batch as _char_save_batch,
)

async def character_list(request):
    return web.json_response({"characters": _char_list()})

async def character_save(request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "需要 JSON body"}, status=400)
    item = _char_save(data)
    if not item:
        return web.json_response({"error": "保存失败"}, status=400)
    return web.json_response({"success": True, "character": item})

async def character_save_batch(request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "需要 JSON body"}, status=400)
    if not isinstance(data, list):
        return web.json_response({"error": "body 必须是一个角色对象数组"}, status=400)
    saved = _char_save_batch(data)
    return web.json_response({"success": True, "saved": saved})

async def character_delete(request):
    try:
        data = await request.json()
    except Exception:
        data = request.query
    row_id = data.get("id") or data.get("name")
    if not row_id:
        return web.json_response({"error": "缺少 id"}, status=400)
    _char_delete(row_id)
    return web.json_response({"success": True})

async def character_by_avatars(request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "需要 JSON body"}, status=400)
    avatars = data.get("avatars", [])
    if not isinstance(avatars, list):
        return web.json_response({"error": "avatars 必须是数组"}, status=400)
    avatars = [str(x) for x in avatars if str(x).strip()]
    return web.json_response({"characters": _char_by_avatars(avatars)})

# ===================== __init__.py JS 预览 API =====================
js_preview_content = None
js_preview_path = os.path.join(os.path.dirname(__file__), "js", "fxai_bottom_preview.js")
if os.path.isfile(js_preview_path):
    try:
        with open(js_preview_path, "r", encoding="utf-8") as f:
            js_preview_content = f.read()
    except Exception:
        pass

async def fxai_bottom_preview(request):
    if js_preview_content is None:
        return web.Response(text="// file not found", content_type="application/javascript", status=404)
    return web.Response(text=js_preview_content, content_type="application/javascript")

# ===================== 统一路由注册 =====================
try:
    # 健康检测
    PromptServer.instance.routes.get("/fxai/health")(health_check)
    PromptServer.instance.routes.get("/fxai/folder/list")(get_folder)
    PromptServer.instance.routes.get("/fxai/files/list")(list_files)
    PromptServer.instance.routes.get("/fxai/files/subdirs")(list_subdirs)
    PromptServer.instance.routes.post("/fxai/files/delete")(delete_file)
    PromptServer.instance.routes.post("/fxai/folders/delete")(delete_folder)
    PromptServer.instance.routes.post("/fxai/folders/create")(create_folder)
    PromptServer.instance.routes.get("/fxai/io/view")(io_preview)
    PromptServer.instance.routes.get("/fxai/text/preview")(text_preview)

    # 模型管理
    PromptServer.instance.routes.get("/fxai/models/subdirs")(list_model_subdirs)
    PromptServer.instance.routes.get("/fxai/models/files")(list_model_files)
    PromptServer.instance.routes.get("/fxai/models/check")(check_model_file)
    PromptServer.instance.routes.post("/fxai/models/check-batch")(check_model_files_batch)
    PromptServer.instance.routes.post("/fxai/models/delete")(delete_model_file)
    PromptServer.instance.routes.post("/fxai/models/clean-empty")(clean_empty_model_dirs)

    # Workflows
    PromptServer.instance.routes.get("/fxai/workflows/list")(list_workflows)
    PromptServer.instance.routes.get("/fxai/workflows/dirs")(list_workflow_dirs)
    PromptServer.instance.routes.get("/fxai/workflows/files")(list_workflow_files)
    PromptServer.instance.routes.post("/fxai/workflows/upload")(upload_workflow)
    PromptServer.instance.routes.post("/fxai/workflows/delete")(delete_workflow)
    PromptServer.instance.routes.get("/fxai/workflows/view")(view_workflow)

    # 系统控制
    PromptServer.instance.routes.post("/fxai/shutdown")(shutdown_pc)
    PromptServer.instance.routes.post("/fxai/reboot")(reboot_pc)
    PromptServer.instance.routes.post("/fxai/restart")(restart_comfyui)

    # 图片管理 - V1
    PromptServer.instance.routes.get("/fxai/image/preview")(image_v1_preview)
    PromptServer.instance.routes.get("/fxai/image/next_number")(image_v1_next_number)
    PromptServer.instance.routes.get("/fxai/image/list")(image_v1_list)
    PromptServer.instance.routes.post("/fxai/image/apply")(image_v1_apply)
    PromptServer.instance.routes.post("/fxai/image/upload")(image_v1_upload)

    # 图片管理 - V2
    PromptServer.instance.routes.get("/fxai/image/v2/preview")(image_v2_preview)
    PromptServer.instance.routes.get("/fxai/image/v2/list")(image_v2_list)
    PromptServer.instance.routes.post("/fxai/image/v2/apply")(image_v2_apply)
    PromptServer.instance.routes.post("/fxai/image/v2/upload")(image_v2_upload)
    PromptServer.instance.routes.delete("/fxai/image/v2/delete")(image_v2_delete)

    # 音频管理
    PromptServer.instance.routes.get("/fxai/audio/preview")(audio_preview)
    PromptServer.instance.routes.get("/fxai/audio/list")(audio_list)
    PromptServer.instance.routes.post("/fxai/audio/apply")(audio_apply)
    PromptServer.instance.routes.post("/fxai/audio/upload")(audio_upload)
    PromptServer.instance.routes.get("/fxai/audio/delete")(audio_delete)

    # 视频管理
    PromptServer.instance.routes.get("/fxai/video/loop/preview")(video_loop_preview)
    PromptServer.instance.routes.get("/fxai/video/list")(video_list)
    PromptServer.instance.routes.post("/fxai/video/apply")(video_apply)
    PromptServer.instance.routes.post("/fxai/video/upload")(video_upload)
    PromptServer.instance.routes.get("/fxai/video/delete")(video_delete)
    PromptServer.instance.routes.get("/fxai/video/preview")(video_preview)

    # 任务查询
    PromptServer.instance.routes.post("/fxai/tasks/result")(task_query)

    # 提示词管理
    PromptServer.instance.routes.get("/fxai/prompt/list")(prompt_list)
    PromptServer.instance.routes.post("/fxai/prompt/save_manual")(prompt_save)
    PromptServer.instance.routes.get("/fxai/prompt/delete")(prompt_delete)

    # 提示词优化
    PromptServer.instance.routes.get("/fxai/prompt/get_models")(prompt_get_models)

    # LoRA 管理
    PromptServer.instance.routes.get("/fxai/lora/files")(lora_files)
    PromptServer.instance.routes.get("/fxai/lora/config")(lora_config)
    PromptServer.instance.routes.get("/fxai/lora/list")(lora_list)

    # 节点更新
    PromptServer.instance.routes.get("/fxai/node/status")(node_status)
    PromptServer.instance.routes.get("/fxai/node/update")(node_update)

    # 音频分段
    PromptServer.instance.routes.get("/fxai/audio-file")(audio_segments_file)
    PromptServer.instance.routes.get("/fxai/audio-waveform")(audio_segments_waveform)
    PromptServer.instance.routes.get("/fxai/audio/segments/file")(audio_segments_v2_file)
    PromptServer.instance.routes.get("/fxai/audio/segments/waveform")(audio_segments_v2_waveform)

    # 角色管理
    PromptServer.instance.routes.get("/fxai/characters/list")(character_list)
    PromptServer.instance.routes.post("/fxai/characters/save")(character_save)
    PromptServer.instance.routes.post("/fxai/characters/save_batch")(character_save_batch)
    PromptServer.instance.routes.post("/fxai/characters/delete")(character_delete)
    PromptServer.instance.routes.post("/fxai/characters/by_avatars")(character_by_avatars)

    # JS 预览
    PromptServer.instance.routes.get("/fxai/image/bottom/preview.js")(fxai_bottom_preview)

except Exception as e:
    print(f"❌ fxai API 挂载失败：{e}")
