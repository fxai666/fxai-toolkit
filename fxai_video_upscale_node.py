# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
"""视频超分放大节点（凤希AI 自研实现，只加载公开的官方权重）。

与市面上的同类节点相比：
  · 支持**任意放大倍数**（1.0~4.0），不写死 2 / 4
  · 输出帧数与输入一致（自动补帧再裁回），不会莫名少几帧
  · 分块按目标分辨率对齐，外边框不羽化，不会压出黑边
"""

import gc
import math

import torch


# ---------------------------------------------------------------------------
# 模型缓存：同一套配置只加载一次权重
# ---------------------------------------------------------------------------
_CACHE = {}


def _free_cache():
    _CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _get_upscaler(dtype, device, use_sparse):
    # 节点文件是被当作顶层模块导入的（fxai-toolkit 目录在 sys.path 上），
    # 所以这里只能写绝对导入，不能用 `from .fxai_vsr...`
    from fxai_vsr.pipeline import VideoUpscaler

    key = (str(dtype), str(device), bool(use_sparse))
    hit = _CACHE.get(key)
    if hit is None:
        _free_cache()
        hit = VideoUpscaler(dtype=dtype, device=device, use_sparse=use_sparse,
                            log=lambda m: print(m))
        _CACHE[key] = hit
    return hit


MIN_OUT = 21          # 流式管线最少能吐出 21 帧（第一块就是 21 帧）


def _frames_for_count(n: int) -> int:
    """补齐到「输出帧数 ≥ n」所需的最少输入帧数。

    流式对齐的账：输入 N 帧 → 对齐 F = largest_8n1(N+4) = 8k+1 → 输出 8k-3 帧。
    所以只有 N 恰好等于 8k-3 时才刚好不丢帧；其余都要补到下一个 8k-3。

    另外 k 必须 ≥ 3（第一个时间块就要 21 帧），所以补帧目标不小于 21 帧。
    需要少于 21 帧时照样补到 21 帧，跑完再从前面裁掉多余的。
    """
    n = max(int(n), 1)
    need = 8 * math.ceil((max(n, MIN_OUT) + 3) / 8) - 3
    return max(need, MIN_OUT)


class FxAiVideoUpscale:
    """把整段画面放大到目标分辨率（生成式超分，非简单插值）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片序列": ("IMAGE", {
                    "tooltip": "待放大的画面序列 (N,H,W,C)，范围 0~1"}),
                "放大倍数": ("FLOAT", {
                    "default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05,
                    "tooltip": "任意倍数。2 倍以上建议配合分块；1.5 倍是「1280×768 → 1920×1152」的常用档"}),
                "分块边长": ("INT", {
                    "default": 0, "min": 0, "max": 2048, "step": 128,
                    "tooltip": "0 = 不分块（最省时但吃显存）。目标像素超过 1280×768 时会被自动设成 384"}),
                "分块重叠": ("INT", {
                    "default": 64, "min": 0, "max": 256, "step": 32,
                    "tooltip": "相邻块的重叠像素，越大接缝越平但越慢"}),
                "色彩校正": ("BOOLEAN", {
                    "default": True, "label_on": "开", "label_off": "关",
                    "tooltip": "用小波把低清原片的色调换回来，避免整体偏色 / 提亮"}),
                "帧数对齐": ("BOOLEAN", {
                    "default": True, "label_on": "对齐", "label_off": "不对齐",
                    "tooltip": "末尾补帧再裁回，保证输出帧数等于输入帧数"}),
                "随机种子": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF, "step": 1}),
                "精度": (["bf16", "fp16"], {"default": "bf16"}),
                "稀疏注意力": ("BOOLEAN", {
                    "default": True, "label_on": "开", "label_off": "关",
                    "tooltip": "块级稀疏注意力。关掉更准但明显更慢"}),
                "卸载模型": ("BOOLEAN", {
                    "default": False, "label_on": "是", "label_off": "否",
                    "tooltip": "跑完把权重从显存里放掉。连续多次放大时别开"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图片序列",)
    FUNCTION = "upscale"
    CATEGORY = "凤希AI/视频"
    DESCRIPTION = ("视频超分放大：用生成式超分把低清画面补出高清细节，"
                   "支持任意倍数（1.0~4.0）与自动分块。权重放 models/FlashVSR/。")

    def upscale(self, 图片序列, 放大倍数=1.5, 分块边长=0, 分块重叠=64,
                色彩校正=True, 帧数对齐=True, 随机种子=0, 精度="bf16",
                稀疏注意力=True, 卸载模型=False):
        import comfy.model_management as mm
        import comfy.utils

        frames = 图片序列
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
        n_in, h_in, w_in, c_in = frames.shape
        print(f"[凤希AI] 视频超分：{n_in} 帧 {w_in}×{h_in} ×{放大倍数:g}")

        device = mm.get_torch_device()
        dtype = torch.float16 if 精度 == "fp16" else torch.bfloat16

        # 目标像素太多又没分块时自动上分块，避免直接 OOM
        want_px = (w_in * 放大倍数) * (h_in * 放大倍数)
        if 分块边长 == 0 and want_px > 1280 * 768:
            分块边长 = 384
            print(f"[凤希AI] 目标 {want_px/1e6:.2f} Mpx 超过 0.98 Mpx，"
                  f"自动启用分块（边长 {分块边长}）")

        up = _get_upscaler(dtype, device, 稀疏注意力)

        # 补帧：让输出帧数足够，跑完再裁回原帧数
        if 帧数对齐:
            need = _frames_for_count(n_in)
            if need > n_in:
                frames = torch.cat(
                    [frames, frames[-1:].repeat(need - n_in, 1, 1, 1)], dim=0)
                print(f"[凤希AI] 补帧 {n_in} → {need}（末帧复制），输出会裁回 {n_in}")

        total = len(frames)
        pbar = comfy.utils.ProgressBar(total)

        def _progress(done, all_):
            try:
                pbar.update_absolute(min(total, int(round(total * done / max(all_, 1)))),
                                     total)
            except Exception:
                pass

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            out = up.upscale(frames, scale=float(放大倍数),
                             tile=(int(分块边长) or None),
                             overlap=int(分块重叠), seed=int(随机种子),
                             color=bool(色彩校正), progress=_progress)
        finally:
            mm.soft_empty_cache()
            if torch.cuda.is_available():
                print(f"[凤希AI] 显存峰值 "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

        if 帧数对齐 and out.shape[0] > n_in:
            out = out[:n_in]
        print(f"[凤希AI] 视频超分完成：{out.shape[0]} 帧 {out.shape[2]}×{out.shape[1]}")

        if 卸载模型:
            _free_cache()
            mm.soft_empty_cache()
            print("[凤希AI] 视频超分权重已从显存释放")

        return (out,)
