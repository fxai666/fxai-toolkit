import os
import server
from aiohttp import web
import urllib.parse

# ==============================================
# 你的原版路由！一行不改！
# ==============================================
async def get_preview(request):
    path = request.query.get("path")
    if not path or not os.path.exists(path):
        return web.Response(status=404)
    return web.FileResponse(path, headers={
        "Content-Type": "video/mp4"
    })

server.PromptServer.instance.routes.get("/fxai/video/preview")(get_preview)

# ==============================================
# 节点：只传数据，不传HTML！
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
        # 只把路径传给前端 JS，不渲染任何HTML
        return {
            "ui": {
                "path": 视频文件路径
            }
        }