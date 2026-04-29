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

# 获取视频目录
def get_video_dir(subdir=""):
    comfy_root = folder_paths.base_path
    base_dir = "fxai/video"
    target_dir = os.path.join(comfy_root, base_dir)
    
    if subdir:
        # 过滤非法字符
        subdir = re.sub(r'[\\/*?:"<>|]', "", subdir)
        target_dir = os.path.join(target_dir, subdir)
    
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

# 列出视频文件
def list_videos(target_dir):
    if not os.path.isdir(target_dir):
        return []
    # 支持的视频格式
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

# 新增：删除单个视频
async def delete_single_video(request):
    try:
        # 获取请求参数
        subdir = request.query.get("subdir", "")
        filename = request.query.get("filename", "")
        
        if not filename:
            return web.json_response({"error": "未提供文件名"}, status=400)
        
        # 安全路径校验
        target_dir = get_video_dir(subdir)
        safe_file = safe_path_join(target_dir, filename)
        
        if not safe_file or not os.path.exists(safe_file):
            return web.json_response({"error": "文件未找到"}, status=404)
        
        # 直接删除文件
        os.remove(safe_file)
        
        # 返回更新后的文件列表
        new_files = list_videos(target_dir)
        return web.json_response({
            "success": True, 
            "files": new_files,
            "message": f"文件 {filename} 已删除"
        })
    except Exception as e:
        return web.json_response({"error": f"删除失败：{str(e)}"}, status=500)

# 清理文件名
def sanitize_filename(filename):
    name = re.sub(r'[\\/*?:"<>|]', '', filename)
    name = name.strip()
    return name

# ---------- HTTP 路由 ----------
# 视频预览
async def get_preview(request):
    subdir = request.query.get("subdir", "")
    filename = request.query.get("filename", "")
    if not filename:
        return web.json_response({"error": "未提供文件名"}, status=400)
    
    target_dir = get_video_dir(subdir)
    safe_file = safe_path_join(target_dir, filename)
    if not safe_file or not os.path.exists(safe_file):
        return web.json_response({"error": "文件未找到"}, status=404)
    
    # 识别MIME类型
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

# 获取视频列表
async def get_file_list(request):
    subdir = request.query.get("subdir", "")
    target_dir = get_video_dir(subdir)
    files = list_videos(target_dir)
    return web.json_response({"files": files, "total": len(files)})

# 应用排序和删除
async def apply_changes(request):
    try:
        data = await request.json()
        subdir = data.get("subdir", "")
        ordered_filenames = data.get("ordered_filenames", [])
        target_dir = get_video_dir(subdir)

        # 获取现有文件
        existing_files = list_videos(target_dir)
        existing_set = set(existing_files)
        # 过滤合法文件
        safe_ordered = [f for f in ordered_filenames if f in existing_set]

        # 删除不在排序列表中的文件
        to_delete = existing_set - set(safe_ordered)
        for f in to_delete:
            fp = safe_path_join(target_dir, f)
            if fp and os.path.exists(fp):
                os.remove(fp)

        # 重命名文件（按排序序号）
        temp_map = []
        for idx, old_fullname in enumerate(safe_ordered):
            old_fp = safe_path_join(target_dir, old_fullname)
            if not old_fp or not os.path.exists(old_fp):
                continue

            # 提取纯文件名（去掉原有序号）
            match = re.match(r'^\d{3}_(.+)', os.path.splitext(old_fullname)[0])
            if match:
                pure_name = match.group(1)
            else:
                pure_name = os.path.splitext(old_fullname)[0]
            
            ext = old_fullname.split('.')[-1].lower()
            new_name = f"{idx:03d}_{pure_name}.{ext}"

            # 临时重命名避免冲突
            temp_name = f"_tmp_{os.urandom(4).hex()}"
            temp_fp = safe_path_join(target_dir, temp_name)
            os.rename(old_fp, temp_fp)
            temp_map.append((temp_fp, new_name))

        # 最终重命名
        for temp_fp, new_name in temp_map:
            final_fp = safe_path_join(target_dir, new_name)
            os.rename(temp_fp, final_fp)

        # 返回新的文件列表
        new_files = list_videos(target_dir)
        return web.json_response({"files": new_files, "success": True})
    except Exception as e:
        return web.json_response({"error": f"应用失败：{str(e)}"}, status=500)

# 视频上传
async def upload_video_custom(request):
    try:
        data = await request.post()
        video = data.get("video")
        subdir = data.get("subdir", "")

        if not video or not hasattr(video, 'file'):
            return web.json_response({"error": "未上传有效视频"}, status=400)

        original_filename = sanitize_filename(video.filename)
        if not original_filename:
            return web.json_response({"error": "文件名为空"}, status=400)

        target_dir = get_video_dir(subdir)
        
        # 获取下一个序号
        file_list = list_videos(target_dir)
        next_num = len(file_list)

        # 生成带序号的文件名
        new_filename = f"{next_num:03d}_{original_filename}"

        # 保存文件
        save_path = safe_path_join(target_dir, new_filename)
        with open(save_path, "wb") as f:
            f.write(video.file.read())
        
        return web.json_response({
            "success": True, 
            "name": new_filename
        })
    except Exception as e:
        return web.json_response({"error": f"上传失败：{str(e)}"}, status=500)

# 注册路由
try:
    server.PromptServer.instance.routes.get("/fxai/video/loop/preview")(get_preview)
    server.PromptServer.instance.routes.get("/fxai/video/list")(get_file_list)
    server.PromptServer.instance.routes.post("/fxai/video/apply")(apply_changes)
    server.PromptServer.instance.routes.post("/fxai/video/upload")(upload_video_custom)
    server.PromptServer.instance.routes.get("/fxai/video/delete")(delete_single_video)
    print("✅ 凤希AI视频资源管理器已就绪")
except Exception as e:
    print(f"❌ 视频管理器启动失败：{e}")

# ComfyUI节点定义
class FxAiVideoManager:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "目录": ("STRING", {"default": "sucai"}),
            },
            "optional": {
                "视频": ("VIDEO", {"forceInput": True}), # 若有VIDEO类型可启用
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("文件夹路径", "视频总数")
    FUNCTION = "run"
    CATEGORY = "凤希AI/视频"

    def run(self, 目录="", 视频=None):
        target_dir = get_video_dir(目录)
        # 若需要保存传入的视频张量，可在这里实现
        files = list_videos(target_dir)
        return (target_dir, len(files))