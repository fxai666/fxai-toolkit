import os
import re
import folder_paths
import server
from aiohttp import web

# 安全路径校验 防止路径穿越漏洞
def safe_path_join(base_dir, path):
    base_dir = os.path.abspath(base_dir)
    full_path = os.path.abspath(os.path.join(base_dir, path))
    if not full_path.startswith(base_dir):
        return None
    return full_path

# 获取提示词根目录
def get_prompt_dir(subdir=""):
    comfy_root = folder_paths.base_path
    base_dir = "fxai/prompts"
    target_dir = os.path.join(comfy_root, base_dir)
    if subdir:
        # 过滤文件名非法字符
        subdir = re.sub(r'[\\/*?:"<>|]', "", subdir)
        target_dir = os.path.join(target_dir, subdir)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

# 列出目录内所有txt提示词文件
def list_prompts(target_dir):
    if not os.path.isdir(target_dir):
        return []
    files = []
    for f in os.listdir(target_dir):
        fp = os.path.join(target_dir, f)
        if os.path.isfile(fp) and f.lower().endswith(".txt"):
            files.append(f)
    files.sort()
    return files

# ------------------- 后端API接口 -------------------
# 获取文件列表
async def get_file_list(request):
    subdir = request.query.get("subdir", "")
    target_dir = get_prompt_dir(subdir)
    files = list_prompts(target_dir)
    return web.json_response({"files": files, "total": len(files)})

# 保存前端自定义输入的提示词（唯一保存接口）
async def save_manual_prompt(request):
    try:
        data = await request.post()
        subdir = data.get("subdir", "")
        filename = data.get("filename", "").strip()
        content = data.get("content", "").strip()

        if not filename:
            return web.json_response({"error": "文件名不能为空"}, status=400)
        if not content:
            return web.json_response({"error": "提示词内容不能为空"}, status=400)

        # 清洗非法字符 + 自动补全txt后缀
        filename = re.sub(r'[\\/*?:"<>|]', "", filename)
        if not filename.lower().endswith(".txt"):
            filename += ".txt"

        target_dir = get_prompt_dir(subdir)
        save_path = safe_path_join(target_dir, filename)
        if not save_path:
            return web.json_response({"error": "非法路径"}, status=403)

        # 写入本地文件
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        return web.json_response({"success": True, "name": filename})
    except Exception as e:
        return web.json_response({"error": f"保存失败：{str(e)}"}, status=500)

# 删除提示词文件
async def delete_prompt(request):
    subdir = request.query.get("subdir", "")
    filename = request.query.get("filename", "")
    if not filename:
        return web.json_response({"error": "未提供文件名"}, status=400)
    
    target_dir = get_prompt_dir(subdir)
    safe_file = safe_path_join(target_dir, filename)
    if not safe_file or not os.path.exists(safe_file):
        return web.json_response({"error": "文件不存在"}, status=404)
    
    try:
        os.remove(safe_file)
        return web.json_response({"success": True, "name": filename})
    except Exception as e:
        return web.json_response({"error": f"删除失败：{str(e)}"}, status=500)

# ------------------- 接口路由注册 -------------------
try:
    server.PromptServer.instance.routes.get("/fxai/prompt/list")(get_file_list)
    server.PromptServer.instance.routes.post("/fxai/prompt/save_manual")(save_manual_prompt)
    server.PromptServer.instance.routes.get("/fxai/prompt/delete")(delete_prompt)
    print("✅ 凤希AI提示词资源管理器已就绪 Q群：775649071")
except Exception as e:
    print(f"❌ 凤希AI提示词资源管理器启动失败：{e}")

# ------------------- ComfyUI 核心节点定义 -------------------
class FxAiPromptManager:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "目录": ("STRING", {"default": "sucai"}),
            },
            "optional":{
                "刷新标记": ("INT", {"forceInput": True}),
            }
        }

    # 修改返回值：添加文件总数输出
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("目录路径", "文件总数")
    FUNCTION = "run"
    CATEGORY = "凤希AI"

    # 运行函数：返回目录路径 + 该目录下txt文件的总数
    def run(self, 目录="",刷新标记=0):
        target_dir = get_prompt_dir(目录)
        file_list = list_prompts(target_dir)  # 获取目录下的txt文件列表
        file_count = len(file_list)          # 计算文件总数
        return (target_dir, file_count,)     # 返回目录路径和总数