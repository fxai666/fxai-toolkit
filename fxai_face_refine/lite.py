# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权

"""
凤希AI - 人脸精修 · 公共层
==========================

**本文件不是节点，是 `lite_v2.py` 依赖的工具库。**

历史：这里原本还有一个 V1 节点（只修一个人，带多参考匹配 / 标号预览 / 开场锁定）。
2026-10-24 按需求下线 —— V2 全脸版接管了全部能力
（所有小脸超分 + 匹配上参考图的每张脸各自换脸 + 保底图）。
V1 的节点类和相关辅助函数已删除，只留 V2 需要的底层实现。

保留的东西
----------
    _t2bgr / _bgr2t / _as_image_batch / _as_image_list   张量 <-> BGR 转换
    _resolve_device                                      显卡编号校验
    _crop_square / _auto_crop / _snap_crop               裁切框（保证 crop × 倍数 = 512）
    _ellipse_mask / _ellipse_mask_hw / _paste            椭圆羽化贴回
    _superres_batch                                      超分（VOSR2 或纯插值）
    _apply_swap / _paste_scaled / _color_match / _inject_texture   换脸合成
    _bc / _bc_done                                       执行状态广播

关键约束（别改）
----------------
`裁切框边长 × 超分倍数` 必须落在 **512**（VOSR2 训练分辨率）：

    裁 128 × 4 = 512   ← 默认，脸 40~85px
    裁 256 × 2 = 512   ← 脸 85~170px
    裁 512 × 1 = 512   ← 脸更大

非原生配比（如 192×3）细节重建明显变差，锐度约降一半。
"""

import os
import time

import cv2
import numpy as np
import torch

from comfy.utils import ProgressBar

from .engines import detect as fx_detect
from .engines import vosr2 as fx_vosr

# 执行状态广播（前端「节点在干什么」靠它）。fxai-toolkit 的兄弟模块，
# 用顶层名导入；万一拿不到就只打日志，绝不影响出图。
try:
    import fxai_task_store
except Exception:       # pragma: no cover - 单独跑脚本时没有这个模块
    fxai_task_store = None

_EVT_START = "face_refine_start"
_EVT_DONE = "face_refine_done"

_INTERP = cv2.INTER_LANCZOS4


def _bc(event, data):
    """发一条过程广播。异常一律吞掉，按无广播处理。"""
    try:
        if fxai_task_store is not None:
            fxai_task_store.broadcast(event, data)
    except Exception as exc:
        print(f"[凤希AI] 广播 {event} 失败（不影响出图）：{exc}")




def _bc_done(result, t0, evt=None):
    """收尾广播：带耗时和报告首行，前端不用等最终输出就知道跑完了。

    evt 可换成别的事件名（V2 节点用自己的一套，前端好区分版本）。
    """
    try:
        rep = ""
        if isinstance(result, (tuple, list)) and len(result) > 1:
            rep = str(result[1])
        head = [ln for ln in rep.splitlines() if ln.strip()][:3]
        dt = time.time() - t0
        msg = f"人脸精修 完成：{dt:.1f}s"
        print(f"[凤希AI] {msg}", flush=True)
        _bc(evt or _EVT_DONE, {
            "message": msg,
            "阶段": "完成",
            "耗时秒": round(dt, 1),
            "摘要": head,
            "报告": rep,
        })
    except Exception as exc:
        print(f"[凤希AI] 收尾广播失败：{exc}")


# ----------------------------------------------------------------------------
# 张量 / 图像转换
# ----------------------------------------------------------------------------
def _t2bgr(t):
    a = t.detach().cpu().numpy()
    if a.ndim == 3 and a.shape[-1] == 4:
        a = a[..., :3]
    a = (np.clip(a, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(a[:, :, ::-1])


def _bgr2t(b):
    return torch.from_numpy(np.ascontiguousarray(b[:, :, ::-1])).float().div_(255.0)


def _as_image_batch(t):
    """把「输入帧」统一成 [B,H,W,C] 张量。

    标准 IMAGE 输出本来就是张量，但有些自家节点输出的是 **tensor 的 list**
    （每张一个 [1,H,W,C]），直接当张量用会报
    'list' object has no attribute 'shape'。这里统一拼一下。
    """
    if t is None:
        return None
    if isinstance(t, (list, tuple)):
        parts = [x if x.dim() == 4 else x[None] for x in t if x is not None]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
    return t


def _as_image_list(t):
    """把「参考图」统一成 [H,W,C] 的列表。

    参考图按列表逐张保留，每张单独处理：几张参考图的尺寸常常不一致
    （1280 和 1024 混用），拼成单个张量会因尺寸不匹配报错。
    """
    if t is None:
        return []
    if isinstance(t, (list, tuple)):
        return [x[0] if x.dim() == 4 else x for x in t if x is not None]
    if int(getattr(t, "shape", [0])[0]) == 0:
        return []
    return [t[i] for i in range(int(t.shape[0]))]


def _resolve_device(device_id):
    """把「显卡编号」落到当前进程真的能用的那张卡上。返回 (可用编号, 提示语)。

    为什么需要这一步
    ----------------
    ComfyUI 在 Windows 上会**强制**把 CUDA_VISIBLE_DEVICES 设成 "0"
    （main.py 里写死的，除非启动时给了 --cuda-device）。所以机器就算插了两张卡，
    节点进程里也只看得见 cuda:0 —— 编号填 1 时 torch 直接抛
    "CUDA error: invalid device ordinal"，而且报错点落在超分引擎加载权重那里
    （vosr2._load_state），光看堆栈根本联想不到是显卡编号的问题。
    同理 insightface 的 prepare(ctx_id=1) 也会失败，于是先打一行
    「人脸检测 GPU 不可用，退回 CPU」，看着像另一个 bug，其实是同一个原因。

    所以这里统一校验一遍：能用的就用，用不了的退回 0 号卡并说清为什么。
    """
    try:
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    if n <= 0:
        return 0, ""
    try:
        want = int(device_id)
    except Exception:
        want = 0
    if 0 <= want < n:
        return want, ""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        why = (f"当前进程只看得见 {n} 张卡（CUDA_VISIBLE_DEVICES={vis}）。\n"
               f"    ComfyUI 在 Windows 上默认锁单卡；真想用第 {want} 张，"
               f"启动时加 --cuda-device all")
    else:
        why = f"当前进程只看得见 {n} 张卡，没有编号 {want} 这张"
    return 0, f"⚠️ 显卡编号 {want} 不可用，已自动改用 0 号卡。\n    {why}"


# ----------------------------------------------------------------------------
# 几何
# ----------------------------------------------------------------------------
def _crop_square(img, face, size):
    """按脸中心裁一块正方形。size 会被限制在画面内并保持偶数（VAE /8 友好）"""
    h, w = img.shape[:2]
    size = int(min(size, h, w))
    size -= size % 2
    cx, cy = face.center
    x0 = max(0, min(int(round(cx - size / 2.0)), w - size))
    y0 = max(0, min(int(round(cy - size / 2.0)), h - size))
    return img[y0:y0 + size, x0:x0 + size].copy(), (x0, y0, x0 + size, y0 + size)


def _auto_crop(max_face_w):
    """按目标脸的宽度自动定裁切框，保证 crop × upscale 正好等于 512。

    框要同时满足两个方向：
    - 框须大于脸：脸比框大会裁进半张脸，下游直接检不到；
    - 框不得远大于脸：脸比框小太多，超分等于把一张小图放大，白花算力还是一样的糊。
    取 1.5 倍脸宽做框，再向上取到 512 的整数分之一，这样放大倍数刚好是 4/2/1。
    """
    need = float(max_face_w) * 1.5
    for c in (128, 256, 512):
        if c >= need:
            return c, fx_vosr.NATIVE // c
    return 512, 1


def _snap_crop(crop_size):
    """把手填的裁切框吸附到 VOSR2 的原生尺寸，并返回配套的放大倍数。

    只有 128 / 256 / 512 三个尺寸能命中 VOSR2 的 latent 对齐
    （patch 16，512/16=32 是 patch 的整数倍）。非原生配比（如 192 x3）会让
    细节重建明显变差，画质掉一档。

    upscale 不由用户填 —— crop × upscale 必须恒等于 512，否则输入尺寸
    偏离训练分布。
    """
    c = int(crop_size)
    for legal in (128, 256, 512):
        if c <= legal:
            return legal, fx_vosr.NATIVE // legal
    return 512, 1


# 标号预览用的配色（BGR）。相邻编号颜色差异要大，缩略图里也分得清。
_PREVIEW_COLORS = [
    (80, 80, 255),     # 红
    (255, 170, 0),     # 蓝
    (0, 200, 0),       # 绿
    (0, 200, 255),     # 黄
    (255, 0, 220),     # 紫
    (0, 120, 255),     # 橙
    (220, 220, 0),     # 青
    (150, 150, 150),   # 灰
]


def _ellipse_mask(size, face_box, mask_scale, feather):
    """椭圆 + 高斯羽化。

    不用 106 点凸包：远景 40px 级的小脸，关键点回归本身就在抖，
    凸包边缘会跟着跳，反而更明显。椭圆稳定得多。
    """
    m = np.zeros((size, size), np.float32)
    x1, y1, x2, y2 = face_box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    ax = max(6.0, (x2 - x1) * mask_scale)
    ay = max(8.0, (y2 - y1) * (mask_scale + 0.04))
    cv2.ellipse(m, (int(round(cx)), int(round(cy))), (int(round(ax)), int(round(ay))),
                0, 0, 360, 1.0, -1)
    sig = max(0.5, size * feather)
    return np.clip(cv2.GaussianBlur(m, (0, 0), sig), 0.0, 1.0)


def _paste(dst, patch, box, mask, strength=1.0):
    x0, y0, x1, y1 = box
    roi = dst[y0:y1, x0:x1]
    if patch.shape[:2] != roi.shape[:2]:
        patch = cv2.resize(patch, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    m = (mask * float(strength))[..., None]
    dst[y0:y1, x0:x1] = (patch.astype(np.float32) * m
                         + roi.astype(np.float32) * (1.0 - m)).round().astype(np.uint8)
    return dst


def _color_match(patch, ref, mask, strength):
    """把换脸后的肤色拉回原脸。

    inswapper 在 128 空间重建整张脸，肤色跟着参考脸走，放到暗光镜头里
    就是一张发白的面具。这里只对齐均值、保持方差不变，原脸的明暗层次得以保留。
    """
    if strength <= 0:
        return patch
    s = patch.astype(np.float32)
    r = ref.astype(np.float32)
    m = mask > 0.5
    if int(m.sum()) < 50:
        return patch
    for c in range(3):
        delta = float(r[..., c][m].mean() - s[..., c][m].mean())
        s[..., c] += delta * float(strength) * mask
    return np.clip(s, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# 超分（引擎调用的薄封装）
# ----------------------------------------------------------------------------
def _superres_batch(engine, crops_bgr, upscale, seed):
    """批量超分。输入 BGR uint8 列表，输出同样格式。"""
    if engine is not None:
        x = torch.from_numpy(np.stack([c[:, :, ::-1] for c in crops_bgr])).float().div_(255.0)
        out = engine.upscale(x, upscale, seed)
        a = (out.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        return [np.ascontiguousarray(v[:, :, ::-1]) for v in a]
    # 没引擎就纯插值放大（不救细节，仅保证链路能走）
    h, w = crops_bgr[0].shape[:2]
    return [cv2.resize(c, (w * upscale, h * upscale), interpolation=_INTERP) for c in crops_bgr]


# ---- 换脸合成（inswapper_128）：由 _apply_swap 统一调用 ----


def _paste_scaled(fake_small, M, dst_bgr, s):
    """按对齐矩阵贴回，但 fake 已被放大 s 倍（边长 = 128*s）。

    M 把 dst 坐标映射到 128 空间；fake_small 在 s 倍空间里，
    所以要先左乘一个 diag(s,s,1) 再求逆。s 取 fake_small 边长 / 128，
    这样 crop_size 不是 128 时也不会错位。
    """
    h, w = dst_bgr.shape[:2]
    M3 = np.vstack([np.asarray(M, np.float32), [0.0, 0.0, 1.0]])
    S3 = np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]], np.float32)
    IM = cv2.invertAffineTransform((S3 @ M3)[:2])
    back = cv2.warpAffine(fake_small, IM, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    cover = cv2.warpAffine(np.full(fake_small.shape[:2], 255, np.uint8), IM, (w, h),
                           flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)
    cover = cv2.erode(cover, np.ones((3, 3), np.uint8), iterations=1)
    out = dst_bgr.copy()
    sel = cover > 0
    out[sel] = back[sel]
    return out, sel


def _inject_texture(swapped, sr_bgr, amount, sigma=1.0):
    """把超分结果的极高频（皮肤锐度、发丝）叠回换脸结果。

    sigma 取小值，只要"最高频"那一层 —— 五官轮廓属于低频，注进来会把
    参考身份冲淡；皮肤质感和发丝属于高频，注进来只提升锐度。
    """
    if amount <= 0:
        return swapped
    detail = sr_bgr.astype(np.float32) - cv2.GaussianBlur(sr_bgr, (0, 0), sigma)
    out = swapped.astype(np.float32) + detail * float(amount)
    return np.clip(out, 0, 255).astype(np.uint8)


def _ellipse_mask_hw(h, w, face_box, mask_scale, feather):
    """整帧尺度的椭圆 mask（非方形画布），换脸合成时用来收边。

    `_ellipse_mask` 只能出 size×size 的正方形 mask（给裁切块用）。
    换脸是直接合成到整帧上的，所以单独给一份 (h, w) 版本。
    """
    m = np.zeros((h, w), np.float32)
    x1, y1, x2, y2 = face_box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    ax = max(6.0, (x2 - x1) * mask_scale)
    ay = max(8.0, (y2 - y1) * (mask_scale + 0.04))
    cv2.ellipse(m, (int(round(cx)), int(round(cy))),
                (int(round(ax)), int(round(ay))), 0, 0, 360, 1.0, -1)
    sig = max(0.5, min(h, w) * feather)
    return np.clip(cv2.GaussianBlur(m, (0, 0), sig), 0.0, 1.0)


def _apply_swap(frame, orig, face, swapper, latent, mask_scale, feather,
                strength, skin_pull, texture_keep):
    """把 inswapper 的结果合成到这张脸的区域，返回整帧。

    顺序有讲究：
      1. 128 结果按对齐矩阵贴回（_paste_scaled 自带方形 cover）
      2. 肤色回拉 —— 不拉的话暗光镜头里会是一张发白的面具
      3. 纹理注入 —— 128 只有 128 空间的信息量，把超分救回来的高频叠回去，
         否则换完比纯超分还糊
      4. 椭圆羽化收边 —— 方形 cover 边缘是硬的，这步把改动限制在脸区内

    frame 是已经超分的整帧，orig 是原帧：5 点对齐用原帧算，
    超分结果的几何与原帧有轻微偏移。
    """
    fake128, M = swapper.algorithm(orig, face.kps, latent)
    swapped, sel = _paste_scaled(fake128, M, frame, 1.0)
    if not sel.any():
        return frame
    h, w = frame.shape[:2]
    ell = _ellipse_mask_hw(h, w, face.bbox, mask_scale, feather)
    swapped = _color_match(swapped, frame, ell, skin_pull)
    swapped = _inject_texture(swapped, frame, texture_keep)
    return _paste(frame.copy(), swapped, (0, 0, w, h), ell, strength)


# ----------------------------------------------------------------------------
# 本模块 = 公共工具层
# ----------------------------------------------------------------------------
# lite.py 只提供裁切 / 超分 / 贴回 / 换脸的工具函数，节点类在 lite_v2.py。
#
# 面板参数用中文名，内部实现用英文名，两者的映射由节点自己维护，不放在这里。
#
# 节点由 fxai-toolkit/__init__.py 的全局注册表统一登记：
#     ("FxAiFaceRestoreLiteV2", "fxai_face_refine", "凤希AI - 人脸精修 · 轻量链 V2（全脸）")
# 全局表用 getattr(module, class_name) 取类，显示名也只在全局表维护一处。
# 各模块不定义 NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS。
