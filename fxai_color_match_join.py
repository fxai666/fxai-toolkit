# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权

"""两段视频按重叠帧做色彩匹配 + 渐变拼接。

分段生成最怕两件事：接缝跳帧、接缝偏色。这里用重叠区的逐通道均值/标准差
把后段线性拉到前段的色调上（增益限幅 0.5~2 倍，防止极端画面把整体压黑），
再在重叠区做 smoothstep 淡化，两段的局部对比度都保留。
"""

import torch


def _ramp(n, device, dtype):
    """n 帧的过渡权重，0→1 的 smoothstep。"""
    x = torch.linspace(0.0, 1.0, n + 2, device=device, dtype=dtype)[1:-1]
    return x * x * (3.0 - 2.0 * x)


class FxAiColorMatchSeamlessJoin:
    """凤希AI - 色彩匹配无缝拼接"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "前段图像": ("IMAGE",),
                "后段图像": ("IMAGE",),
                "重叠帧": ("INT", {"default": 5, "min": 0, "max": 340, "step": 1,
                                 "tooltip": "两段之间重叠的帧数，后段的前 N 帧会被淡化融进前段尾部"}),
            },
            "optional": {
                "色彩匹配": ("BOOLEAN", {"default": True}),
                "匹配强度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "1.0 = 完全对齐前段色调；调低可以留一点后段原本的氛围"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "凤希AI/视频"

    def run(self, 前段图像, 后段图像, 重叠帧, 色彩匹配=True, 匹配强度=1.0):
        a, b = 前段图像, 后段图像

        if a.shape[1:] != b.shape[1:]:
            raise ValueError("两段尺寸不一致：%s vs %s" % (tuple(a.shape[1:3]), tuple(b.shape[1:3])))
        if b.dtype != a.dtype:
            b = b.to(a.dtype)

        ov = max(0, min(重叠帧, a.shape[0], b.shape[0]))
        s = float(匹配强度) if 色彩匹配 else 0.0

        if ov > 0 and s > 0.0:
            ref, tgt = a[-ov:], b[:ov]
            m_ref, s_ref = ref.mean(dim=(0, 1, 2)), ref.std(dim=(0, 1, 2))
            m_tgt, s_tgt = tgt.mean(dim=(0, 1, 2)), tgt.std(dim=(0, 1, 2))
            gain = (s_ref / s_tgt.clamp_min(1e-5)).clamp(0.5, 2.0)
            bias = m_ref - gain * m_tgt
            b = (b * (1.0 + (gain - 1.0) * s) + bias * s).clamp(0.0, 1.0)
            print("[色彩匹配拼接] 增益 %s  偏移 %s"
                  % (["%.3f" % g for g in gain.tolist()], ["%.3f" % x for x in bias.tolist()]))

        if ov > 0:
            w = _ramp(ov, a.device, a.dtype).view(ov, 1, 1, 1)
            joint = a[-ov:] * (1.0 - w) + b[:ov] * w
            return (torch.cat([a[:-ov], joint, b[ov:]], dim=0),)

        return (torch.cat([a, b], dim=0),)
