import os
import re
import time
import shutil
import folder_paths
import subprocess

def safe_path_join(base_dir, path):
    base_dir = os.path.abspath(base_dir)
    full_path = os.path.abspath(os.path.join(base_dir, path))
    return full_path if full_path.startswith(base_dir) else None

def get_merge_output_dir():
    comfy_root = folder_paths.base_path
    target_dir = os.path.join(comfy_root, "fxai/video/merged")
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

# ====================== 改动 1：增加 max_count 参数 ======================
def get_video_files(source_dir, max_count=0):
    if not os.path.isdir(source_dir):
        return []
    exts = ('.mp4', '.webm', '.mov', '.avi')
    files = sorted(f for f in os.listdir(source_dir) if f.lower().endswith(exts))
    
    # 核心约束：只保留前 max_count 个文件
    if max_count > 0:
        files = files[:max_count]
        
    return [safe_path_join(source_dir, f) for f in files]

def merge_videos(source_dir, output_name, max_count=0):
    # ====================== 改动 2：传入文件数量 ======================
    videos = get_video_files(source_dir, max_count)
    output_dir = get_merge_output_dir()
    
    output_name = re.sub(r'[\\/*?:"<>|]', "", output_name.strip())
    output_path = safe_path_join(output_dir, f"{output_name}.mp4")

    if len(videos) == 0:
        print("[凤希AI视频合并] 无视频")
        return None
    elif len(videos) == 1:
        shutil.copy2(videos[0], output_path)
    else:
        list_path = os.path.join(source_dir, "merge_list.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for p in videos:
                f.write(f"file '{p}'\n")
        cmd = f'ffmpeg -y -f concat -safe 0 -i "{list_path}" -c copy "{output_path}"'
        subprocess.run(cmd, shell=True, check=True)
        os.remove(list_path)
    return output_path

class FxAiVideoMerger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "源视频文件夹路径": ("STRING", {"default": ""}),
                "文件数量": ("INT", {"default": 1,"step":1}),
                "名称前缀": ("STRING", {"default": "fxai_"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("视频本地路径",)
    FUNCTION = "run"
    CATEGORY = "凤希AI/视频"

    def run(self, 源视频文件夹路径, 名称前缀,文件数量=0):
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        final_name = f"{名称前缀}{time_str}"

        if not os.path.isdir(源视频文件夹路径):
            return ("",)
        
        # ====================== 改动 3：把文件数量传给合并函数 ======================
        video_path = merge_videos(源视频文件夹路径, final_name, 文件数量)
        if not video_path:
            return ("",)
        
        return (video_path,)