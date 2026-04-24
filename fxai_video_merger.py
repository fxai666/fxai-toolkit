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

def get_video_files(source_dir):
    if not os.path.isdir(source_dir):
        return []
    exts = ('.mp4', '.webm', '.mov', '.avi')
    files = sorted(f for f in os.listdir(source_dir) if f.lower().endswith(exts))
    return [safe_path_join(source_dir, f) for f in files]

def merge_videos(source_dir, output_name):
    videos = get_video_files(source_dir)
    output_dir = get_merge_output_dir()
    
    output_name = re.sub(r'[\\/*?:"<>|]', "", output_name.strip())
    output_path = safe_path_join(output_dir, f"{output_name}.mp4")

    if len(videos) == 0:
        print("[凤希合并] 无视频")
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
                "名称前缀": ("STRING", {"default": "fxai_"}),
            },
            "optional":{
                "刷新标记": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("视频本地路径",)
    FUNCTION = "run"
    CATEGORY = "凤希AI"

    def run(self, 源视频文件夹路径, 名称前缀,刷新标记=0):
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        final_name = f"{名称前缀}{time_str}"

        if not os.path.isdir(源视频文件夹路径):
            return ("",)
        
        video_path = merge_videos(源视频文件夹路径, final_name)
        if not video_path:
            return ("",)
        
        return (video_path,)