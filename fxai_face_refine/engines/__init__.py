# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License

"""
凤希AI 人脸精修 · 引擎层
========================

两个引擎都是自研实现，不依赖 `custom_nodes/ComfyUI-VOSR2`：

    vosr2  生成式超分：把糊掉的五官重建出来（权重用 VOSR 团队训练的那个 DiT）
    detect 检测 + ROI 跟踪：找脸、取五点、算身份向量

模型权重都是公开下载的，代码全部自己写。

`swap.py`（换脸，hyperswap_1a_256 / inswapper_128）是可选引擎：模块级只 import
os/threading/numpy，onnxruntime 在 `SwapEngine.__init__` 里才 lazy import，所以没装
onnxruntime 或两个换脸权重都没有时也不影响超分链路启动。
"""

from . import detect, paths, vosr2  # noqa: F401

try:                       # 换脸是可选能力，导入失败不该连累精修主链路
    from . import swap  # noqa: F401
except Exception as _swap_exc:  # pragma: no cover
    print(f"⚠️[凤希AI] 换脸引擎导入失败（不影响精修）: {_swap_exc}")
    swap = None

__all__ = ["vosr2", "detect", "paths", "swap"]
