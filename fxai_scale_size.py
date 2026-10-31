# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权

"""凤希AI - 宽高-按百分比缩放

通用尺寸换算器：把「源素材宽高」按百分比缩放成「生成用宽高」。

存在的理由
----------
生成模型只认「对齐后的画布」，而素材千奇百怪。每次换素材都要手算
「原片 1920×1080 缩 90% = 1728×972，对齐 32 变 1728×960」，算一次错一次。
这个节点把这一步固化：

    源宽高 × 百分比 → 按对齐基数向下取整 → 直接喂生成控制器

对齐规则与 FxAIGeneratorController 完全一致（基数 × 采样次数，floor），
所以喂进控制器后数值不会再变 —— 控制器里填多少，模型就用多少。

源尺寸从哪来
------------
三种都支持，优先级从高到低：

    1. 分辨率（STRING）  ← 接「凤希AI - 视频加载」的「分辨率」输出（"1920x1080"）
    2. 原宽度 / 原高度   ← 接「视频加载」的「视频宽度/视频高度」，
                          或「视频分段器V1」的「原始宽度/原始高度」
    3. 手动填数字        ← 什么都没接时的兜底
"""

import re


# ---------------------------------------------------------------- 纯数值计算

def parse_resolution(text):
    """把 "1920x1080" / "1920×1080" / "1920 * 1080" 解析成 (w, h)，失败返回 None。"""
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None
    m = re.search(r"(\d+)\s*[xX×*]\s*(\d+)", s)
    if not m:
        return None
    w, h = int(m.group(1)), int(m.group(2))
    if w <= 0 or h <= 0:
        return None
    return w, h


def scale_size(src_w, src_h, percent, base=32, samples=1):
    """按百分比缩放宽高，并按「对齐基数 × 采样次数」向下取整。

    与 FxAIGeneratorController 的算法逐字一致：
        有效基数 = 对齐基数 × 采样次数
        最终宽 = 宽 // 有效基数 * 有效基数

    百分比先四舍五入到像素，再 floor 对齐 —— 只做一次取整，
    避免「先 floor 再 floor」把小尺寸越缩越小。
    """
    pct = float(percent)
    if pct <= 0:
        raise ValueError(f"百分比必须大于 0，当前 {percent}")

    eff = max(2, int(base)) * max(1, int(samples))

    w = int(round(int(src_w) * pct / 100.0))
    h = int(round(int(src_h) * pct / 100.0))

    # 极小素材缩放后可能掉到 0，至少保一个对齐单位
    w = max(eff, w)
    h = max(eff, h)

    fw = (w // eff) * eff
    fh = (h // eff) * eff
    return fw, fh, eff


def _fmt_pct(pct):
    return f"{pct:g}%"


# ---------------------------------------------------------------- 节点

class FxAiScaleSize:
    """宽高-按百分比缩放（源素材尺寸 → 生成画布尺寸）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "百分比": ("FLOAT", {
                    "default": 90.0, "min": 1.0, "max": 400.0, "step": 0.5,
                    "display": "number",
                    "tooltip": "相对源素材的缩放比例。100 = 原尺寸，90 = 缩到九成。\n"
                               "注意：H3 原生画布面积上限约 1.03 Mpx（1344×768），\n"
                               "超过这个面积模型会自己再压，等于白算。",
                }),
                "对齐基数": ("INT", {
                    "default": 32, "min": 2, "max": 128, "step": 2,
                    "tooltip": "与生成控制器的「长宽对齐基数」填同一个数（默认 32）。\n"
                               "实际取整基数是「对齐基数 × 采样次数」。",
                }),
                "采样次数": ("INT", {
                    "default": 1, "min": 1, "max": 4, "step": 1,
                    "tooltip": "与生成控制器的「采样次数」填同一个数。\n"
                               "单采 = 1（基数 32）；双采 = 2（基数 64，否则控制器还会再砍一刀）。",
                }),
            },
            "optional": {
                "原宽度": ("INT", {
                    "default": 1920, "min": 16, "max": 16384, "step": 2,
                    "display": "number",
                    "tooltip": "源素材宽度。可接「视频加载」的「视频宽度」或\n"
                               "「视频分段器V1」的「原始宽度」。",
                }),
                "原高度": ("INT", {
                    "default": 1080, "min": 16, "max": 16384, "step": 2,
                    "display": "number",
                    "tooltip": "源素材高度。",
                }),
                "分辨率": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "可选。接「视频加载」的「分辨率」输出（如 1920x1080）。\n"
                               "接了就以它为准，忽略上面手填的宽高。",
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("宽度", "高度", "半宽", "半高", "对齐基数", "缩放信息")
    FUNCTION = "process"
    CATEGORY = "凤希AI/工具"

    def process(self, 百分比, 对齐基数=32, 采样次数=1,
                原宽度=1920, 原高度=1080, 分辨率=""):
        parsed = parse_resolution(分辨率)
        if parsed:
            sw, sh = parsed
            src_note = "来自分辨率输入"
        else:
            sw, sh = int(原宽度), int(原高度)
            src_note = "来自原宽高"

        if sw <= 0 or sh <= 0:
            raise RuntimeError(f"源尺寸不合法：{sw}×{sh}")

        fw, fh, eff = scale_size(sw, sh, 百分比, 对齐基数, 采样次数)

        half_w = (fw // 2 // max(2, int(对齐基数))) * max(2, int(对齐基数))
        half_h = (fh // 2 // max(2, int(对齐基数))) * max(2, int(对齐基数))
        half_w = max(int(对齐基数), half_w)
        half_h = max(int(对齐基数), half_h)

        info = (f"{sw}×{sh} ×{_fmt_pct(百分比)} → {fw}×{fh}"
                f"（对齐 {eff}，{src_note}）")

        print(f"📐 [凤希AI] 宽高缩放：{info}")

        return (fw, fh, half_w, half_h, eff, info)
