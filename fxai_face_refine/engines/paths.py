# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License

"""
凤希AI 模型路径解析
===================

三个引擎都要找 `models/` 下的东西（vosr2 / insightface），各写一份探测逻辑
迟早会不一致，所以统一到这里。

踩过的坑：脚本环境里 `import folder_paths` 是能成功的（ComfyUI 根目录在
sys.path 上），但这时 ComfyUI 还没跑初始化，`models_dir` 指向一个并不存在
的位置。检测器拿着这个错路径去找 buffalo_l，找不到就去 GitHub 下载 ——
表现为「跑着跑着卡在下载模型了」。

所以这里不盲目相信 `folder_paths`：拿到路径后要验证它像不像真的 models 目录，
不像就退回到按 `__file__` 往上数的方式。
"""

import os


def _looks_like_models(p):
    """真正的 ComfyUI models 目录下应该有这些常见子目录之一。"""
    if not p or not os.path.isdir(p):
        return False
    return any(os.path.isdir(os.path.join(p, d))
               for d in ("checkpoints", "vae", "loras", "clip", "diffusion_models"))


def comfy_root():
    """定位 ComfyUI 根目录：从本文件往上数 5 层。

    .../custom_nodes/fxai-toolkit/fxai_face_refine/engines/paths.py
         └────────────┬───────────────────────────────┘
          1: engines  2: fxai_face_refine  3: fxai-toolkit
          4: custom_nodes  5: <ComfyUI 根>
    """
    here = os.path.abspath(__file__)
    root = here
    for _ in range(5):
        root = os.path.dirname(root)
    return root


def models_root():
    """按优先级找 models 目录。"""
    env = os.environ.get("FXAI_MODELS_DIR")
    if env and os.path.isdir(env):
        return env

    try:
        import folder_paths
        p = getattr(folder_paths, "models_dir", None)
        if _looks_like_models(p):
            return p
    except Exception:
        pass

    cand = os.path.join(comfy_root(), "models")
    if os.path.isdir(cand):
        return cand

    # 最后兜底：向上逐层找带 checkpoints 的 models 目录
    root = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        c = os.path.join(root, "models")
        if _looks_like_models(c):
            return c
        root = os.path.dirname(root)
    return cand


def vosr2_dir(bundle="VOSR2"):
    return os.path.join(models_root(), "vosr2", bundle)


def insightface_root():
    return os.path.join(models_root(), "insightface")


def inswapper_path():
    return os.path.join(insightface_root(), "models", "inswapper_128.onnx")


def buffalo_dir():
    return os.path.join(insightface_root(), "models", "buffalo_l")
