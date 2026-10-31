# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权

"""凤希AI - 宽高-按百分比缩放

纯换算器：源宽高 × 百分比 → 两个整数宽高，输出给下游。

这里只做乘法，不做对齐、不切半宽、不猜下游要什么基数 —— 对齐规则在
FxAIGeneratorController 里（有效基数 = 长宽对齐基数 × 采样次数，floor），
在别处再算一遍只会算错，还会跟控制器打架。

源尺寸从哪来
------------
三种都支持，优先级从高到低：

    1. 分辨率（STRING）  ← 接「视频加载」的「分辨率」输出（"1920x1080"）
    2. 原宽度 / 原高度   ← 接「视频加载」的「视频宽度/视频高度」，
                          或「视频分段器V1」的「原始宽度/原始高度」
    3. 手动填数字        ← 什么都没接时的兜底
"""

import re


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


def scale_size(src_w, src_h, percent):
    """源宽高 × 百分比，四舍五入到整数像素。"""
    pct = float(percent)
    if pct <= 0:
        raise ValueError(f"百分比必须大于 0，当前 {percent}")
    w = int(round(int(src_w) * pct / 100.0))
    h = int(round(int(src_h) * pct / 100.0))
    # 缩得极狠时可能掉到 0，下游拿到 0 会直接崩，兜到 1
    return max(1, w), max(1, h)


class FxAiScaleSize:
    """宽高-按百分比缩放（源素材尺寸 → 生成画布尺寸）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "百分比": ("FLOAT", {
                    "default": 100.0, "min": 1.0, "max": 400.0, "step": 0.5,
                    "display": "number",
                    "tooltip": "相对源素材的缩放比例。100 = 原尺寸，90 = 缩到九成。\n"
                               "注意：H3 原生画布面积上限约 1.03 Mpx（1344×768），\n"
                               "超过这个面积模型会自己再压，等于白算。",
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

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("宽度", "高度")
    FUNCTION = "process"
    CATEGORY = "凤希AI/工具"

    def process(self, 百分比, 原宽度=1920, 原高度=1080, 分辨率=""):
        parsed = parse_resolution(分辨率)
        if parsed:
            sw, sh = parsed
            src_note = "来自分辨率输入"
        else:
            sw, sh = int(原宽度), int(原高度)
            src_note = "来自原宽高"

        if sw <= 0 or sh <= 0:
            raise RuntimeError(f"源尺寸不合法：{sw}×{sh}")

        w, h = scale_size(sw, sh, 百分比)
        print(f"📐 [凤希AI] 宽高缩放：{sw}×{sh} ×{百分比:g}% → {w}×{h}（{src_note}）")
        return (w, h)
