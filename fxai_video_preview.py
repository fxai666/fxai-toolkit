import os
import server
from aiohttp import web
from datetime import datetime

# ==============================================
# ✅ 精简版：纯字母数字文件名，无需编码
# ==============================================
async def get_preview(request):
    path = request.query.get("path")
    if not path or not os.path.exists(path):
        return web.Response(status=404)
    
    # 生成时间戳文件名：fxai_年月日时分秒.mp4
    time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"fxai_{time_str}.mp4"

    # 直接用纯文件名，无编码
    headers = {
        "Content-Type": "video/mp4",
        "Content-Disposition": f'inline; filename="{filename}"'
    }

    return web.FileResponse(path, headers=headers)

server.PromptServer.instance.routes.get("/fxai/video/preview")(get_preview)

# ==============================================
# 节点：不变
# ==============================================
class FxAiVideoPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "视频文件路径": ("STRING", {"default": ""}),
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "凤希AI"

    def run(self, 视频文件路径):
        return {
            "ui": {
                "path": 视频文件路径
            }
        }