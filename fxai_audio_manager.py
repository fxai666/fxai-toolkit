import os
import re
import torch
import folder_paths
import server
from aiohttp import web
import mimetypes
import soundfile as sf
import numpy as np

# 安全路径校验：防止目录穿越
def safe_path_join(base_dir, path):
    base_dir = os.path.abspath(base_dir)
    full_path = os.path.abspath(os.path.join(base_dir, path))
    if not full_path.startswith(base_dir):
        return None
    return full_path

def get_audio_dir(subdir=""):
    comfy_root = folder_paths.base_path
    base_dir = "fxai/audio"
    target_dir = os.path.join(comfy_root, base_dir)
    
    if subdir:
        subdir = re.sub(r'[\\/*?:"<>|]', "", subdir)
        target_dir = os.path.join(target_dir, subdir)
    
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

def list_audios(target_dir):
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
            files.append((m.group(1), f))
    files.sort()
    return [f for _, f in files]

# ---------- HTTP 路由 ----------
async def get_preview(request):
    subdir = request.query.get("subdir", "")
    filename = request.query.get("filename", "")
    if not filename:
        return web.json_response({"error": "未提供文件名"}, status=400)
    
    target_dir = get_audio_dir(subdir)
    safe_file = safe_path_join(target_dir, filename)
    if not safe_file or not os.path.exists(safe_file):
        return web.json_response({"error": "文件未找到"}, status=404)
    
    mime_type = mimetypes.guess_type(safe_file)[0]
    if not mime_type:
        ext = filename.split('.')[-1].lower()
        mime_map = {
            'mp3': 'audio/mpeg',
            'wav': 'audio/wav',
            'ogg': 'audio/ogg',
            'flac': 'audio/flac',
            'm4a': 'audio/mp4'
        }
        mime_type = mime_map.get(ext, "audio/mpeg")
    
    return web.FileResponse(safe_file, headers={
        "Content-Type": mime_type,
        "Cache-Control": "no-store, no-cache, must-revalidate"
    })

async def get_next_number(request):
    subdir = request.query.get("subdir", "")
    target_dir = get_audio_dir(subdir)
    used = set()
    if os.path.isdir(target_dir):
        for f in os.listdir(target_dir):
            m = re.match(r'^(\d+)', f)
            if m:
                used.add(int(m.group(1)))
    next_num = 0
    while next_num in used:
        next_num += 1
    return web.json_response({"next_num": next_num})

async def get_file_list(request):
    subdir = request.query.get("subdir", "")
    target_dir = get_audio_dir(subdir)
    files = list_audios(target_dir)
    return web.json_response({"files": files, "total": len(files)})

async def apply_changes(request):
    try:
        data = await request.json()
        subdir = data.get("subdir", "")
        ordered_filenames = data.get("ordered_filenames", [])
        target_dir = get_audio_dir(subdir)

        existing_files = list_audios(target_dir)
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

        new_files = list_audios(target_dir)
        return web.json_response({"files": new_files, "success": True})
    except Exception as e:
        return web.json_response({"error": f"应用失败：{str(e)}"}, status=500)

async def upload_audio_custom(request):
    try:
        data = await request.post()
        audio = data.get("audio")
        subdir = data.get("subdir", "")

        if not audio or not hasattr(audio, 'file'):
            return web.json_response({"error": "未上传有效音频"}, status=400)

        filename = re.sub(r'[\\/*?:"<>|]', "", audio.filename)
        if not filename:
            return web.json_response({"error": "文件名为空"}, status=400)

        target_dir = get_audio_dir(subdir)
        save_path = safe_path_join(target_dir, filename)
        if not save_path:
            return web.json_response({"error": "非法路径"}, status=403)

        with open(save_path, "wb") as f:
            f.write(audio.file.read())

        return web.json_response({"success": True, "name": filename})
    except Exception as e:
        return web.json_response({"error": f"上传失败：{str(e)}"}, status=500)

# 注册路由
try:
    server.PromptServer.instance.routes.get("/fxai/audio/preview")(get_preview)
    server.PromptServer.instance.routes.get("/fxai/audio/next_number")(get_next_number)
    server.PromptServer.instance.routes.get("/fxai/audio/list")(get_file_list)
    server.PromptServer.instance.routes.post("/fxai/audio/apply")(apply_changes)
    server.PromptServer.instance.routes.post("/fxai/audio/upload")(upload_audio_custom)
    print("✅ 凤希AI音频资源管理器已就绪 Q群：775649071")
except Exception as e:
    print(f"❌ 凤希AI音频资源管理器启动失败：{e}")

class FxAiAudioManager:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "目录": ("STRING", {"default": "sucai"}),
            },
            "optional":{
                "音频": ("AUDIO", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "AUDIO")
    RETURN_NAMES = ("文件列表", "文件夹路径", "音频总数", "音频")
    FUNCTION = "run"
    CATEGORY = "凤希AI"

    def _get_next_audio_num(self, target_dir):
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

    def save_tensor_audio(self, audio_data, target_dir):
        try:
            # 读取 ComfyUI 标准音频
            waveform = audio_data["waveform"]
            sample_rate = audio_data["sample_rate"]

            # 🔥 修复采样率（张量 → 数字）
            if isinstance(sample_rate, torch.Tensor):
                sample_rate = sample_rate.item()

            audio_np = waveform.cpu().numpy()

            # 维度标准化 100% 兼容 soundfile
            if audio_np.ndim == 3:
                audio_np = audio_np[0]
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(-1, 1)
            if audio_np.ndim == 2 and audio_np.shape[0] in (1, 2):
                audio_np = audio_np.transpose(1, 0)

            audio_np = np.squeeze(audio_np)
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(-1, 1)

            # 保存
            next_num = self._get_next_audio_num(target_dir)
            save_path = os.path.join(target_dir, f"{next_num:03d}.wav")
            sf.write(save_path, audio_np, sample_rate)
            print(f"✅ 音频保存成功：{save_path}")

        except Exception as e:
            print(f"❌ 保存音频失败：{str(e)}")
            raise

    def run(self, 目录="", 音频=None):
        target_dir = get_audio_dir(目录)
        
        if 音频 is not None:
            self.save_tensor_audio(音频, target_dir)
        
        files = list_audios(target_dir)
        total = len(files)
        file_str = "\n".join(files) if files else "无音频，请先上传"
        
        return (file_str, target_dir, total, 音频)