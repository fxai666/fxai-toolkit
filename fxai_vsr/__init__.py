# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
"""凤希AI 视频超分 —— 自研实现。

模型结构、流式管线、分块策略全部自己写，只加载公开的官方权重，
因此可以做到官方节点做不到的：
  · 任意放大倍数（含 1.5 倍）—— 官方节点写死 2 / 4
  · 分块时外边框不羽化，不会有黑边
  · 模型常驻显存，不重复加载
"""

from .dit import DiTConfig, StreamDiT
from .lq import LQProjector
from .tcdecoder import TCDecoder
from .loader import build_models, model_root
from .pipeline import VideoUpscaler

__all__ = [
    "DiTConfig", "StreamDiT", "LQProjector", "TCDecoder",
    "build_models", "model_root", "VideoUpscaler",
]
