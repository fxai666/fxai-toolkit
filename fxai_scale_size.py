# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权

"""凤希AI - 宽高-按最大边长缩放

把「源素材宽高」压进一个方形上限里，只输出两个整数宽高。

规则一句话：最长边不超过「最大边长」。
    源片最大边 ≤ 最大边长  → 原样输出，不放大
    源片最大边 > 最大边长  → 等比缩到最大边正好等于上限

比按百分比省事：不用知道素材是横是竖、多大，填一个 1280 就够，
1080×1904 的竖片和 1920×1080 的横片都会落在 1280 这条线内。

这里只做乘除法，不做对齐、不切半宽、不猜下游要什么基数 —— 对齐规则在
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


def fit_max_side(src_w, src_h, max_side):
    """最长边压到 max_side 以内，等比缩放；本来就够小就原样返回。"""
    src_w, src_h = int(src_w), int(src_h)
    max_side = int(max_side)
    if max_side <= 0:
        raise ValueError(f"最大边长必须大于 0，当前 {max_side}")

    longest = max(src_w, src_h)
    if longest <= max_side:
        return src_w, src_h

    scale = max_side / longest
    w = max(1, int(round(src_w * scale)))
    h = max(1, int(round(src_h * scale)))
    return w, h


class FxAiScaleSize:
    """宽高-按最大边长缩放（源素材尺寸 → 生成画布尺寸）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "最大边长": ("INT", {
                    "default": 1280, "min": 64, "max": 16384, "step": 2,
                    "display": "number",
                    "tooltip": "最长边不超过这个值。\n"
                               "源片最大边已经小于它 → 原样输出，不放大；\n"
                               "大于它 → 等比缩到最大边正好等于它。\n"
                               "注意：H3 原生画布面积上限约 1.03 Mpx（1344×768），\n"
                               "1280 已经贴近这条线，再往上模型会自己压，等于白算。",
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

    def process(self, 最大边长, 原宽度=1920, 原高度=1080, 分辨率=""):
        parsed = parse_resolution(分辨率)
        if parsed:
            sw, sh = parsed
            src_note = "来自分辨率输入"
        else:
            sw, sh = int(原宽度), int(原高度)
            src_note = "来自原宽高"

        if sw <= 0 or sh <= 0:
            raise RuntimeError(f"源尺寸不合法：{sw}×{sh}")

        w, h = fit_max_side(sw, sh, 最大边长)
        keep = "，未超过上限原样输出" if (w, h) == (sw, sh) else ""
        print(f"📐 [凤希AI] 宽高缩放：{sw}×{sh} 最长边 {max(sw, sh)} "
              f"→ 上限 {最大边长} → {w}×{h}（{src_note}{keep}）")
        return (w, h)
