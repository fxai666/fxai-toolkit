import os
import re
import torch
import folder_paths
import server
from aiohttp import web
import mimetypes
import soundfile as sf
import numpy as np
import subprocess
import json

# 安全路径校验
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
            files.append(f)
    files.sort()
    return files

def sanitize_filename(filename):
    name = re.sub(r'[\\/*?:"<>|]', '', filename)
    name = name.strip()
    return name

# ========== 重点：纯ffprobe获取时长，移除torchaudio依赖 ==========
def get_audio_duration(audio_path):
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            audio_path
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10
        )
        meta = json.loads(result.stdout)
        duration = float(meta["format"]["duration"])
        return round(duration, 3)
    except Exception:
        return 0.0

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
            'mp3': 'audio/mpeg', 'wav': 'audio/wav', 'ogg': 'audio/ogg',
            'flac': 'audio/flac', 'm4a': 'audio/mp4'
        }
        mime_type = mime_map.get(ext, "audio/mpeg")
    
    return web.FileResponse(safe_file, headers={
        "Content-Type": mime_type,
        "Cache-Control": "no-store, no-cache, must-revalidate"
    })

async def get_file_list(request):
    subdir = request.query.get("subdir", "")
    target_dir = get_audio_dir(subdir)
    files = list_audios(target_dir)
    # 【兼容旧前端】依然返回files字段，不破坏原有逻辑
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
        for idx, old_fullname in enumerate(safe_ordered):
            old_fp = safe_path_join(target_dir, old_fullname)
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
            temp_fp = safe_path_join(target_dir, temp_name)
            os.rename(old_fp, temp_fp)
            temp_map.append((temp_fp, new_name))

        for temp_fp, new_name in temp_map:
            final_fp = safe_path_join(target_dir, new_name)
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

        original_filename = sanitize_filename(audio.filename)
        if not original_filename:
            return web.json_response({"error": "文件名为空"}, status=400)

        target_dir = get_audio_dir(subdir)

        file_list = list_audios(target_dir)
        next_num = len(file_list)

        # 原始文件先落盘（保留原后缀），再统一转成 wav
        base_name = os.path.splitext(original_filename)[0]
        raw_path = safe_path_join(target_dir, original_filename)
        with open(raw_path, "wb") as f:
            f.write(audio.file.read())

        new_filename = f"{next_num:03d}_{base_name}.wav"
        wav_path = safe_path_join(target_dir, new_filename)
        cmd = [
            "ffmpeg", "-i", raw_path,
            "-f", "wav", "-y", wav_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)

        # 转换成功删除原始文件；转换失败保留原始文件兜底
        if os.path.exists(wav_path):
            try:
                os.remove(raw_path)
            except Exception:
                pass
        else:
            new_filename = original_filename

        return web.json_response({
            "success": True,
            "name": new_filename
        })
    except Exception as e:
        return web.json_response({"error": f"上传失败：{str(e)}"}, status=500)

async def delete_single_audio(request):
    try:
        subdir = request.query.get("subdir", "")
        filename = request.query.get("filename", "")
        
        if not filename:
            return web.json_response({"error": "未提供文件名"}, status=400)
        
        target_dir = get_audio_dir(subdir)
        safe_file = safe_path_join(target_dir, filename)
        
        if not safe_file or not os.path.exists(safe_file):
            return web.json_response({"error": "文件未找到"}, status=404)
        
        os.remove(safe_file)
        
        return web.json_response({
            "success": True
        })
    except Exception as e:
        return web.json_response({"error": f"删除失败：{str(e)}"}, status=500)

# 注册路由
try:
    server.PromptServer.instance.routes.get("/fxai/audio/preview")(get_preview)
    server.PromptServer.instance.routes.get("/fxai/audio/list")(get_file_list)
    server.PromptServer.instance.routes.post("/fxai/audio/apply")(apply_changes)
    server.PromptServer.instance.routes.post("/fxai/audio/upload")(upload_audio_custom)
    server.PromptServer.instance.routes.get("/fxai/audio/delete")(delete_single_audio)
except Exception as e:
    print(f"❌ 启动失败：{e}")

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

    # ==================== 【完全原样保留，一丝不改】 ====================
    RETURN_TYPES = ("STRING", "INT", "AUDIO")
    RETURN_NAMES = ("文件夹路径", "音频总数", "音频")
    FUNCTION = "run"
    CATEGORY = "凤希AI/音频"

    def save_tensor_audio(self, audio_data, target_dir):
        try:
            waveform = audio_data["waveform"]
            sample_rate = audio_data["sample_rate"]
            if isinstance(sample_rate, torch.Tensor):
                sample_rate = sample_rate.item()

            audio_np = waveform.cpu().numpy()
            if audio_np.ndim == 3: audio_np = audio_np[0]
            if audio_np.ndim == 1: audio_np = audio_np.reshape(-1, 1)
            if audio_np.shape[0] in (1,2): audio_np = audio_np.transpose(1,0)
            audio_np = np.squeeze(audio_np)
            if audio_np.ndim == 1: audio_np = audio_np.reshape(-1,1)

            next_num = len(list_audios(target_dir))
            save_name = f"{next_num:03d}_audio.wav"
            save_path = os.path.join(target_dir, save_name)
            sf.write(save_path, audio_np, sample_rate)
        except Exception as e:
            print(f"❌ 保存音频失败：{e}")

    def run(self, 目录="", 音频=None,):
        target_dir = get_audio_dir(目录)
        if 音频 is not None:
            self.save_tensor_audio(音频, target_dir)
        
        files = list_audios(target_dir)

        return (target_dir, len(files), 音频)
		
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")