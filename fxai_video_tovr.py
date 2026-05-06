import os
import re
import subprocess
import torch
import folder_paths
import gc

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpeg', '.mpg', '.webm')

def safe_path_join(base_dir, path):
    base_dir = os.path.abspath(base_dir)
    full_path = os.path.abspath(os.path.join(base_dir, path))
    if not full_path.startswith(base_dir):
        return None
    return full_path

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

def get_video_dir(subdir=""):
    comfy_root = folder_paths.base_path
    base_dir = "fxai/video"
    target_dir = os.path.join(comfy_root, base_dir)
    if subdir:
        subdir = re.sub(r'[\\/*?:"<>|]', "", subdir)
        target_dir = os.path.join(target_dir, subdir)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

def get_ffmpeg_path():
    try:
        ffmpeg_path = folder_paths.get_full_path("ffmpeg", "ffmpeg")
        if ffmpeg_path and os.path.exists(ffmpeg_path):
            return ffmpeg_path
    except Exception:
        pass
    return "ffmpeg"

# ==============================================
# 凤希AI 视频转VR —— 完全还原你的原版
# 只修复180°黄色画面, 无任何强制分辨率, 无拉伸
# ==============================================
class FxAiVideoToVR:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "视频文件夹路径": ("STRING", {"default": ""}),
                "视频索引": ("INT", {"default": 0, "min": 0}),
                "保存目录": ("STRING", {"default": "VR"}),
                "视频序号": ("INT", {"default": 0, "min": 0}),
                "VR模式": ([
                    "左右分屏SBS",
                    "上下分屏OU",
                    "180°沉浸",
                    "360°影院"
                ], {"default": "180°沉浸"}),
                "输出宽度": ("INT", {"default": 4096, "min": 512, "max": 16384}),
                "帧率FPS": ("INT", {"default": 24, "min": 1}),
                "编码方式": (["CPU-x264", "GPU-NVENC", "HEVC-x265"], {"default": "CPU-x264"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("VR视频路径", "保存目录")
    FUNCTION = "convert_vr"
    CATEGORY = "凤希AI/视频"

    def convert_vr(self, 视频文件夹路径, 视频索引, 保存目录, 视频序号, VR模式, 输出宽度, 帧率FPS, 编码方式):
        vr_path = ""
        save_dir = ""

        try:
            folder = 视频文件夹路径.strip()
            if not os.path.isdir(folder):
                raise Exception("视频文件夹不存在")

            files = sorted([f for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXTENSIONS)])
            if not files:
                raise Exception("文件夹内无视频")

            if 视频索引 < 0 or 视频索引 >= len(files):
                raise Exception("视频索引越界")

            src = os.path.join(folder, files[视频索引])
            save_dir = get_video_dir(保存目录)
            num = 视频序号 if 视频序号 > 0 else get_last_number(save_dir)
            fname = f"{num:03d}_vr.mp4"
            vr_path = safe_path_join(save_dir, fname)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 编码器
            if 编码方式 == "GPU-NVENC":
                vcodec = ["-c:v", "hevc_nvenc", "-preset", "p4", "-qp", "23"]
            elif 编码方式 == "HEVC-x265":
                vcodec = ["-c:v", "libx265", "-crf", "20"]
            else:
                vcodec = ["-c:v", "libx264", "-crf", "18"]

            ffmpeg_bin = get_ffmpeg_path()
            cmd = [ffmpeg_bin, "-y", "-i", src, "-r", str(帧率FPS), "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"] + vcodec

            w = 输出宽度
            h = w // 2

            # ===================== 完全还原你原版逻辑！无任何强制分辨率！=====================
            if VR模式 == "左右分屏SBS":
                cmd += [
                    "-filter_complex",
                    f"[0:v]split=2[l][r];[l][r]hstack=inputs=2,scale={w}:{h}",
                    vr_path
                ]
            elif VR模式 == "上下分屏OU":
                cmd += [
                    "-filter_complex",
                    f"[0:v]scale={w}:{h//2},split=2[t][b];[t][b]vstack=inputs=2",
                    vr_path
                ]
            elif VR模式 == "180°沉浸":
                cmd += [
                    "-vf",
                    f"v360=input=flat:output=e:ih_fov=180:iv_fov=180:w={w}:h={h},format=yuv420p",
                    "-metadata:s:v", "projection=equirectangular",
                    vr_path
                ]
            elif VR模式 == "360°影院":
                cmd += [
                    "-vf",
                    f"v360=input=flat:output=e:w={w}:h={h},format=yuv420p",
                    "-metadata:s:v", "projection=equirectangular",
                    vr_path
                ]

            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
            if proc.returncode != 0:
                raise Exception(f"FFmpeg错误：{proc.stderr[-500:]}")

            print(f"[凤希AI VR] 转换成功：{vr_path}")

        except Exception as e:
            print(f"[凤希AI VR 错误] {str(e)}")
            vr_path = save_dir = ""

        return (vr_path, save_dir)