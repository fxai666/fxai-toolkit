# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# 接缝色彩对齐：把本段整体平移到「上一段末帧」的色阶上。
#
# 为什么需要它：
#   分段生成时每一段都是独立从噪声跑出来的，模型对「整屏曝光/色温」没有跨段记忆。
#   实测（608x1056 / 216 帧 / 两段）接缝处整屏亮度会掉 1.6~2.0 个灰阶，
#   而全片正常的帧间亮度变化 95 分位只有 0.6~0.7 —— 也就是说接缝处是正常变化的
#   近 3 倍。帧差指标看不出来（被大量本来就在微动的像素稀释了），但眼睛一眼就看见。
#
# 做法：逐通道求「参考帧均值 − 本段首帧均值」，把整个本段加上这个差值。
#   只平移、不缩放，所以对比度和色彩关系不变；整段统一加同一个值，
#   段内不会引入新的跳变。本段末尾跟着一起平移，正好成为下一段的参考，
#   整条片子于是锁在第一段的色阶上，误差不累积。
#
# 偏移量有上限（默认 12 灰阶）：万一参考帧本身是闪白/黑场，不会把整段带跑。

import torch


class FxAiSeamColorMatch:
    CATEGORY = "凤希AI/视频"
    FUNCTION = "run"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图片序列",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片序列": ("IMAGE", {
                    "tooltip": "本段解码+裁剪之后的帧序列。"}),
            },
            "optional": {
                "参考帧": ("IMAGE", {
                    "tooltip": "上一段的末帧（接缝锚点）。不接就原样透传，"
                               "第一段本来就没有上一段。"}),
                "最大偏移": ("FLOAT", {
                    "default": 12.0, "min": 0.0, "max": 40.0, "step": 0.5,
                    "tooltip": "单位是灰阶（0-255）。对齐量超过这个值就按上限截断，\n"
                               "防止参考帧是闪白/黑场时把整段带偏。\n"
                               "填 0 等于关闭本节点。"}),
            },
        }

    def run(self, 图片序列, 参考帧=None, 最大偏移=12.0):
        if 图片序列 is None or 图片序列.shape[0] == 0:
            return (图片序列,)

        limit = float(最大偏移 or 0.0) / 255.0
        if 参考帧 is None or 参考帧.shape[0] == 0 or limit <= 0.0:
            return (图片序列,)

        ref = 参考帧[-1:].to(torch.float32)
        cur = 图片序列[:1].to(torch.float32)

        # 逐通道均值差（0-1 空间），(3,)
        delta = ref.mean(dim=(0, 1, 2)) - cur.mean(dim=(0, 1, 2))
        raw = delta.clone()
        delta = delta.clamp(-limit, limit)

        if bool((delta.abs() < 1e-6).all()):
            return (图片序列,)

        out = (图片序列.to(torch.float32) + delta.view(1, 1, 1, 3)).clamp_(0.0, 1.0)
        out = out.to(图片序列.dtype)

        print("[凤希AI·接缝] 色彩对齐：需要 %+.2f %+.2f %+.2f 灰阶，实际施加 %+.2f %+.2f %+.2f"
              % (raw[0] * 255, raw[1] * 255, raw[2] * 255,
                 delta[0] * 255, delta[1] * 255, delta[2] * 255))
        return (out,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
