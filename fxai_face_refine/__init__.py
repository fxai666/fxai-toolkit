# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权

"""
凤希AI - H3 人脸精修（子包）
============================

专治 MiniMax H3 远景小脸：头在画面里占比太小时，latent 分不到足够 token，
脸就糊、就变形。靠放大救不回来（放大只会把错的脸放得更清楚）。

做法是逐帧处理：
    检测人脸 -> 按人脸中心逐帧裁剪并归一化到 768 canvas -> 交给 H3 带参考图
    重绘 -> 带羽化和色彩匹配贴回原片

目录结构：
    fxai_face_refine/
        __init__.py      本文件：导出节点类 + 注册后端路由
        nodes.py         7 个节点主体
        picker_api.py    Pick faces 选择器的后端路由
    js/face_refine/
        fxai_h3_face_select.js   前端「Pick faces」对话框

用法要点（配合双采工作流）：
    二采出图 -> 本模块修脸 -> 可选收尾增强
    修脸必须排在放大之前：canvas 固定 768，脸已经大于 768 时再修就是缩小，
    magnification < 1.0 会丢真实像素，越修越差。
"""

from .nodes import (
    H3FaceSelect,
    H3FaceTrackCrop,
    H3FaceStitch,
    H3InjectVideoLatent,
    H3PerFrameDenoise,
    H3FaceMaskSAM,
    H3FaceTransformInfo,
)

# 轻量链只有 V2 全脸版一个节点：多参考自动认人、兜底、换脸都在它里面。
# lite.py 是它的工具库，只导出函数。
try:
    from .lite_v2 import FxAiFaceRestoreLiteV2
except Exception as _v2_exc:  # pragma: no cover
    print(f"❌[凤希AI] 轻量链 V2 加载失败: {_v2_exc}")
    FxAiFaceRestoreLiteV2 = None


def register():
    """注册人脸选择器后端路由。

    路由：/h3_facerefine/videos | preview | preview_xy | scan

    选择器在整张图开始跑之前就要出结果，所以它读不到节点产物，
    而是自己重跑一遍 加载/检测/分镜，返回每个镜头一张带编号人脸的卡片。

    这里做容错：路由挂不上不该连累节点本身加载。
    """
    try:
        from . import picker_api
        picker_api.register()
        return True
    except Exception as _exc:
        print(f"❌[凤希AI] 人脸选择器路由注册失败: {_exc}")
        return None


__all__ = [
    "H3FaceSelect",
    "H3FaceTrackCrop",
    "H3FaceStitch",
    "H3InjectVideoLatent",
    "H3PerFrameDenoise",
    "H3FaceMaskSAM",
    "H3FaceTransformInfo",
    "FxAiFaceRestoreLiteV2",
    "register",
]
