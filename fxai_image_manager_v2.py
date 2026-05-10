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

# =============== 空文件夹返回 0，完全正常 ===============
def get_next_number(target_dir):
    files = list_images(target_dir)
    return len(files)

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

async def get_file_list(request):
    subdir = request.query.get("subdir", "")
    target_dir = get_image_dir(subdir)
    files = list_images(target_dir)
    return web.json_response({"files": files, "total": len(files)})

# ===================== 【唯一修改的地方：仅这里！】 =====================
async def apply_changes(request):
    try:
        data = await request.json()
        subdir = data.get("subdir", "")
        ordered_filenames = data.get("ordered_filenames", [])
        target_dir = get_image_dir(subdir)

        if not ordered_filenames:
            return web.json_response({"files": list_images(target_dir), "success": True, "msg": "无需修改"})

        # 第一步：直接加序号前缀，绝对不重名
        for idx, old_name in enumerate(ordered_filenames):
            old_path = safe_path_join(target_dir, old_name)
            if not old_path or not os.path.exists(old_path):
                continue
            temp_name = f"{idx:03d}_{old_name}"
            temp_path = safe_path_join(target_dir, temp_name)
            os.rename(old_path, temp_path)

        # 第二步：精简成 000.png
        for idx, old_name in enumerate(ordered_filenames):
            temp_name = f"{idx:03d}_{old_name}"
            temp_path = safe_path_join(target_dir, temp_name)
            if not temp_path or not os.path.exists(temp_path):
                continue
            # 直接取后缀，不解析、不复杂处理
            ext = old_name.split(".")[-1]
            final_name = f"{idx:03d}.{ext}"
            final_path = safe_path_join(target_dir, final_name)
            os.rename(temp_path, final_path)

        new_files = list_images(target_dir)
        return web.json_response({"files": new_files, "success": True})
    except Exception as e:
        return web.json_response({"error": f"应用失败：{str(e)}"}, status=500)
# ===================== 【修改结束】 =====================

async def upload_image_custom(request):
    try:
        data = await request.post()
        image = data.get("image")
        subdir = data.get("subdir", "")

        if not image or not hasattr(image, 'file'):
            return web.json_response({"error": "未上传有效图片"}, status=400)

        original_filename = re.sub(r'[\\/*?:"<>|]', "", image.filename)
        if not original_filename:
            return web.json_response({"error": "文件名为空"}, status=400)

        target_dir = get_image_dir(subdir)
        next_num = get_next_number(target_dir)
        
        ext = original_filename.split('.')[-1].lower()
        if ext not in ['png', 'jpg', 'jpeg', 'webp']:
            ext = 'png'
        
        # =============== 网页上传永远自动计数，不受文件名序号影响 ===============
        new_filename = f"{next_num:03d}.{ext}"
        save_path = safe_path_join(target_dir, new_filename)

        with open(save_path, "wb") as f:
            f.write(image.file.read())

        return web.json_response({"success": True, "name": new_filename})
    except Exception as e:
        return web.json_response({"error": f"上传失败：{str(e)}"}, status=500)

async def delete_image(request):
    try:
        subdir = request.query.get("subdir", "")
        filename = request.query.get("filename", "")
        if not filename:
            return web.json_response({"error": "未提供文件名"}, status=400)
        
        target_dir = get_image_dir(subdir)
        safe_file = safe_path_join(target_dir, filename)
        
        if not safe_file or not os.path.exists(safe_file):
            return web.json_response({"error": "文件不存在"}, status=404)
        
        os.remove(safe_file)
        new_files = list_images(target_dir)
        return web.json_response({"success": True, "files": new_files})
    except Exception as e:
        return web.json_response({"error": f"删除失败：{str(e)}"}, status=500)

# 注册路由
try:
    server.PromptServer.instance.routes.get("/fxai/image/v2/preview")(get_preview)
    server.PromptServer.instance.routes.get("/fxai/image/v2/list")(get_file_list)
    server.PromptServer.instance.routes.post("/fxai/image/v2/apply")(apply_changes)
    server.PromptServer.instance.routes.post("/fxai/image/v2/upload")(upload_image_custom)
    server.PromptServer.instance.routes.delete("/fxai/image/v2/delete")(delete_image)
    print("✅ 凤希AI图片资源管理器已就绪 Q群：775649071")
except Exception as e:
    print(f"❌ 凤希AI图片资源管理器启动失败：{e}")

class FxAiImageManagerV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "目录": ("STRING", {"default": "sucai"}),
            },
            "optional":{
                "图片": ("IMAGE", {"forceInput": True}),
                "刷新标记": ("INT", {"forceInput": True}),
                "文件名序号": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "IMAGE", "INT")
    RETURN_NAMES = ("文件夹路径", "图片总数", "图片", "刷新标记")
    FUNCTION = "run"
    CATEGORY = "凤希AI/图片"

    def save_tensor_image(self, image_tensor, save_dir, custom_num=None):
        if image_tensor is None or not isinstance(image_tensor, torch.Tensor):
            return
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # 仅节点传入图片张量时：有文件名序号就用，没有就自动计数
            if custom_num is not None and isinstance(custom_num, int) and custom_num >= 0:
                start_num = custom_num
            else:
                start_num = get_next_number(save_dir)
            
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            for i in range(image_np.shape[0]):
                img = Image.fromarray(image_np[i])
                filename = f"{start_num + i:03d}.png"
                save_path = os.path.join(save_dir, filename)
                img.save(save_path, format="PNG")
                print(f"[凤希AI图片资源管理] 已保存：{save_path}")
        except Exception as e:
            print(f"[凤希AI图片资源管理] 保存失败：{e}")

    def run(self, 目录="", 图片=None, 刷新标记=0,文件名序号=None):
        target_dir = get_image_dir(目录)
        
        # 只有左侧输入图片时，才使用文件名序号
        if 图片 is not None:
            self.save_tensor_image(图片, target_dir, custom_num=文件名序号)
        
        files = list_images(target_dir)
        
        return (target_dir, len(files), 图片,刷新标记)