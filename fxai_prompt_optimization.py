import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import requests
from aiohttp import web

try:
    from server import PromptServer
except:
    PromptServer = None

# 默认空提示
_DEFAULT_MODEL = "huihui_ai/qwen3.5-abliterated:9b"

# ------------------------------
# 异步接口
# ------------------------------
async def api_get_ollama_models(request):
    host = request.query.get("host", "http://127.0.0.1:11434")
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{host}/api/tags", timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m["name"] for m in data.get("models", [])]
                    return web.json_response({"models": models if models else [_DEFAULT_MODEL]})
    except:
        pass
    return web.json_response({"models": [_DEFAULT_MODEL]})

if PromptServer:
    try:
        PromptServer.instance.routes.get("/fxai/prompt/get_models")(api_get_ollama_models)
    except:
        pass

# ------------------------------
# 节点主体
# ------------------------------
class FxAiPromptGenerator:
    CATEGORY = "凤希AI"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "是否开启提示词优化": ("BOOLEAN", {"default": True,"tooltip":"当关闭时，原样输出提示词"}),
                "API主机地址": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "模型选择": ([_DEFAULT_MODEL], {"default": _DEFAULT_MODEL}),
                "推理后释放资源": ("BOOLEAN", {"default": True}),
                "系统提示词": ("STRING", {
                    "multiline": True
                }),
            },
            "optional": {
                "分段时长": ("FLOAT", {"default": 0, "forceInput": True}),
                "用户提示词": ("STRING", {
                    "default": "请根据图片生成优质AI绘画提示词",
                    "multiline": True,
					"forceInput": True
                }),
                "图片一": ("IMAGE",),
                "图片二": ("IMAGE",),
                "图片三": ("IMAGE",),
            }
        }
    @classmethod
    def VALIDATE_INPUTS(cls, 模型选择, **kwargs):
        return True

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("生成的提示词", "分段时长")
    FUNCTION = "generate"

    def generate(self,是否开启提示词优化, API主机地址, 模型选择, 推理后释放资源, 系统提示词,分段时长=0,用户提示词="", 图片一=None, 图片二=None, 图片三=None):
        if not 是否开启提示词优化:
           return (用户提示词, 分段时长)

        if 模型选择 == "":
            return ("⚠️ 请先点击【刷新模型】加载 Ollama 模型",)

        images = []
        if 图片一 is not None:
            images.append(self.t2b64(图片一))
        if 图片二 is not None:
            images.append(self.t2b64(图片二))
        if 图片三 is not None:
            images.append(self.t2b64(图片三))

        # 最终返回内容，默认 = 原始用户提示词
        final_output = 用户提示词

        try:
            if 分段时长 > 0:
                系统提示词 = f"请您根据要求生成一段总长度{分段时长}秒的视频脚本；{系统提示词}"

            # 模型参数使用 模型选择
            resp = requests.post(f"{API主机地址}/api/generate", json={
                "model": 模型选择,
                "system": 系统提示词,
                "prompt": 用户提示词,
                "images": images,
                "stream": False
            })

            if resp.status_code == 200:
                res_text = resp.json().get("response", "").strip()
                if res_text:
                    final_output = res_text

            if 推理后释放资源:
                try:
                    requests.post(f"{API主机地址}/api/generate", json={"model": 模型选择, "keep_alive": 0}, timeout=3)
                except:
                    pass

        except Exception as e:
            pass

        return (final_output, 分段时长)

    def t2b64(self, t):
        if len(t.shape) == 4:
            t = t.squeeze(0)
        i = Image.fromarray((t.cpu().numpy() * 255).astype(np.uint8))
        b = BytesIO()
        i.save(b, "PNG")
        return base64.b64encode(b.getvalue()).decode()