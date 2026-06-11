import os
import re
import folder_paths
from aiohttp import web
from server import PromptServer

def get_folder(request):
    # 从请求参数获取 subdir
    subdir = request.query.get("subdir", "")

    comfy_root = folder_paths.base_path
    safe_sub = re.sub(r'[\\/*?:"<>|]', "", subdir.strip())
    target_dir = os.path.join(comfy_root, "fxai", safe_sub)

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

try:
    PromptServer.instance.routes.get("/fxai/folder/list")(get_folder)
except Exception as e:
    print(f"❌ 凤希AI目录查询接口挂载失败：{e}")