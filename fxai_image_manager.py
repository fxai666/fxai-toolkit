import os
import re
import torch
import numpy as np
from PIL import Image
import folder_paths
import server
from aiohttp import web
import mimetypes

# 安全路径校验：防止目录穿越
def safe_path_join(base_dir, path):
    base_dir = os.path.abspath(base_dir)
    full_path = os.path.abspath(os.path.join(base_dir, path))
    if not full_path.startswith(base_dir):
        return None
    return full_path

def get_image_dir(subdir=""):
    comfy_root = folder_paths.base_path
    base_dir = "fxai/image"
    target_dir = os.path.join(comfy_root, base_dir)
    
    if subdir:
        subdir = re.sub(r'[\\/*?:"<>|]', "", subdir)
        target_dir = os.path.join(target_dir, subdir)
    
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

def get_last_number(target_dir):
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

def list_images(target_dir):
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

def load_image(file_path):
    try:
        img = Image.open(file_path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except:
        return None

# ---------- HTTP 路由 ----------
async def get_preview(request):
    subdir = request.query.get("subdir", "")
    filename = request.query.get("filename", "")
    if not filename:
        return web.json_response({"error": "未提供文件名"}, status=400)
    
    target_dir = get_image_dir(subdir)
    safe_file = safe_path_join(target_dir, filename)
    if not safe_file or not os.path.exists(safe_file):
        return web.json_response({"error": "文件未找到"}, status=404)
    
    return web.FileResponse(safe_file, headers={
        "Content-Type": mimetypes.guess_type(safe_file)[0] or "image/png",
        "Cache-Control": "no-store, no-cache, must-revalidate"
    })

async def get_next_number(request):
    subdir = request.query.get("subdir", "")
    target_dir = get_image_dir(subdir)
    next_num = get_last_number(target_dir)
    return web.json_response({"next_num": next_num})

async def get_file_list(request):
    subdir = request.query.get("subdir", "")
    target_dir = get_image_dir(subdir)
    files = list_images(target_dir)
    return web.json_response({"files": files, "total": len(files)})

async def apply_changes(request):
    try:
        data = await request.json()
        subdir = data.get("subdir", "")
        ordered_filenames = data.get("ordered_filenames", [])
        target_dir = get_image_dir(subdir)

        existing_files = list_images(target_dir)
        existing_set = set(existing_files)
        safe_ordered = [f for f in ordered_filenames if f in existing_set]

        to_delete = existing_set - set(safe_ordered)
        for f in to_delete:
            fp = safe_path_join(target_dir, f)
            if fp:
                os.remove(fp)

        temp_map = []
        for idx, old_name in enumerate(safe_ordered):
            old_fp = safe_path_join(target_dir, old_name)
            if not old_fp or not os.path.exists(old_fp):
                continue
            
            ext = old_name.split('.')[-1].lower()
            new_name = f"{idx:03d}.{ext}"
            temp_name = f"_tmp_{idx}_{os.urandom(4).hex()}_{old_name}"
            temp_fp = safe_path_join(target_dir, temp_name)
            
            os.rename(old_fp, temp_fp)
            temp_map.append((temp_fp, new_name))

        for temp_fp, new_name in temp_map:
            final_fp = safe_path_join(target_dir, new_name)
            if temp_fp and final_fp:
                os.rename(temp_fp, final_fp)

        new_files = list_images(target_dir)
        return web.json_response({"files": new_files, "success": True})
    except Exception as e:
        return web.json_response({"error": f"应用失败：{str(e)}"}, status=500)

async def upload_image_custom(request):
    try:
        data = await request.post()
        image = data.get("image")
        subdir = data.get("subdir", "")

        if not image or not hasattr(image, 'file'):
            return web.json_response({"error": "未上传有效图片"}, status=400)

        # 安全过滤文件名
        original_filename = re.sub(r'[\\/*?:"<>|]', "", image.filename)
        if not original_filename:
            return web.json_response({"error": "文件名为空"}, status=400)

        target_dir = get_image_dir(subdir)
        
        # ✅ 自动获取当前最大序号 + 1
        next_num = get_last_number(target_dir)
        
        # ✅ 保留原始后缀！！！
        ext = original_filename.split('.')[-1].lower()
        if ext not in ['png', 'jpg', 'jpeg', 'webp']:
            ext = 'png'
        
        new_filename = f"{next_num:03d}.{ext}"
        save_path = safe_path_join(target_dir, new_filename)

        # 保存
        with open(save_path, "wb") as f:
            f.write(image.file.read())

        return web.json_response({"success": True, "name": new_filename})
    except Exception as e:
        return web.json_response({"error": f"上传失败：{str(e)}"}, status=500)

# 注册路由
try:
    server.PromptServer.instance.routes.get("/fxai/image/preview")(get_preview)
    server.PromptServer.instance.routes.get("/fxai/image/next_number")(get_next_number)
    server.PromptServer.instance.routes.get("/fxai/image/list")(get_file_list)
    server.PromptServer.instance.routes.post("/fxai/image/apply")(apply_changes)
    server.PromptServer.instance.routes.post("/fxai/image/upload")(upload_image_custom)
    print("✅ 凤希AI图片资源管理器已就绪 Q群：775649071")
except Exception as e:
    print(f"❌ 凤希AI图片资源管理器启动失败：{e}")

class FxAiImageManager:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "目录": ("STRING", {"default": "sucai"}),
            },
            "optional":{
                "图片": ("IMAGE", {"forceInput": True}),
                "刷新标记": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "IMAGE", "INT")
    RETURN_NAMES = ("文件列表", "文件夹路径", "图片总数", "图片", "刷新标记")
    FUNCTION = "run"
    CATEGORY = "凤希AI/图片"

    def save_tensor_image(self, image_tensor, save_dir):
        if image_tensor is None or not isinstance(image_tensor, torch.Tensor):
            return
        try:
            os.makedirs(save_dir, exist_ok=True)
            next_num = get_last_number(save_dir)
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            for i in range(image_np.shape[0]):
                img = Image.fromarray(image_np[i])
                filename = f"{next_num + i:03d}.png"
                save_path = os.path.join(save_dir, filename)
                img.save(save_path, format="PNG")
                print(f"[凤希AI图片资源] 已保存：{save_path}")
        except Exception as e:
            print(f"[凤希AI图片资源] 保存失败：{e}")

    def run(self, 目录="", 图片=None, 刷新标记=0):
        target_dir = get_image_dir(目录)
        
        if 图片 is not None:
            self.save_tensor_image(图片, target_dir)
        
        files = list_images(target_dir)
        total = len(files)
        file_str = "\n".join(files) if files else "无图片，请先上传"
        
        return (file_str, target_dir, total, 图片,刷新标记)