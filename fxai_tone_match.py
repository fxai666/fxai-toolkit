# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# 影调对齐：把生成段的背景亮度拉回到源片的水准。
#
# 为什么需要它：
#   角色替换跑完之后整片会「发闷」——不是加了暗角，也不是高光被压，
#   而是整屏背景被均匀压低了约 9~11 个灰阶。实测（608x1056 / 216 帧）：
#     四角背景  86.5 → 75.2   （-11.3）
#     径向剖面  r>0.36 之后差值是一条水平线（-8 ~ -17），不随半径加深 → 无暗角
#     时间走势  首 20 帧 -9.41 / 末 20 帧 -9.14，漂移仅 +0.27 → 不漂移
#   既然既不是暗角也不漂移，就不需要逐帧自适应：一个固定量就够，也不会抖。
#
# 为什么不用提示词：
#   往正向提示词里写 "keeps the bright, evenly lit look / background brightness
#   and contrast stay identical"，2643 字符版实跑下来四角 75.2 一个灰阶都没动。
#   反过来写 no darkening / no vignette 会让画面更暗（均值 79 → 59）——文本编码器
#   读不懂否定，等于反向指挥。影调只能靠后处理，不能靠说。
#
# 为什么不直接复用「接缝色彩对齐」：
#   那个节点求的是整帧均值差，而整帧均值被人物服装主导（白上衣换黑蕾丝，
#   中央区差 -46），会算出远超实际的偏移量。这里只采样背景区。
#
# 为什么要卡增益范围（2026-10-31 加）：
#   暗背景片子上会算出灾难性的比值。实测某段生成四角 70/3/6/11、源片 29/1/5/2，
#   四块比 0.411/0.335/0.923/0.173，剔离群后 0.2986 —— 整段亮度砍到 30%，
#   对比度从 44 塌到 13.6，画面像蒙了一层灰雾。
#   前两段没事是因为有一块 84 的亮区被当离群值剔掉了；这一段四块全暗，
#   没有离群值可剔，四个噪声级的小比值直接平均成了灾难。
#   真实漂移实测只有 12%（86.5 → 75.2，比值 0.87），所以：
#     比值落在 [0.7, 1.4] 之外 → 判定采样不可靠，本段跳过，画面不动。
#   宁可少调一段，也不能把整段压成灰雾。
#
# 为什么取中位数而不是平均值：
#   长裙、阔腿裤这类会伸进画面下两角，把「四角」的下两个采样块从背景变成衣服。
#   实测穿及地汉服时，下两角比源片高 +14/+5，上两角却是 -7/-3 —— 取平均会让
#   真正的背景角欠校正 5~7 个灰阶。取中位数后，一两个角被侵占会自动被剔除。
#
# 采样区怎么选：
#   八方位 —— 默认。3×3 网格去掉中心（中心是人物），8 块一起估，最稳
#   四角   —— 4 个角块。能扛住 1~2 个角被人物侵占
#   上部   —— 只有上两角。及地长裙/阔腿裤挡满下半幅时用这个最稳
#   边缘带 —— 上下左右四条边带的中位数，适合人物占满左右的镜头
#   全图   —— 只在整帧都是空镜时用，有人物时会被服装带偏

import torch

# 增益的合理区间。实测漂移约 12%（比值 0.87），±40% 已经远超正常范围；
# 超出去说明采样不可靠（暗背景、闪白、构图突变），宁可不调
GAIN_MIN, GAIN_MAX = 0.7, 1.4


def _patch_means(gray, region, k):
    """gray: [H, W] float32 → 该区域各采样块的平均亮度（0-1）列表。"""
    h, w = gray.shape
    if region == "全图":
        return [gray.mean()]

    kk = min(k, h // 2, w // 2)
    if region == "八方位":
        # 3×3 网格去掉正中那块：中间一定是人物，采进来会被服装主导
        b = min(k, h // 3, w // 3)
        cy, cx = h // 2, w // 2
        half = b // 2
        blocks = [
            gray[:b, :b],
            gray[:b, cx - half:cx + half],
            gray[:b, w - b:],
            gray[cy - half:cy + half, :b],
            gray[cy - half:cy + half, w - b:],
            gray[h - b:, :b],
            gray[h - b:, cx - half:cx + half],
            gray[h - b:, w - b:],
        ]
        return [blk.mean() for blk in blocks]

    tl = gray[:kk, :kk].mean()
    tr = gray[:kk, w - kk:].mean()
    bl = gray[h - kk:, :kk].mean()
    br = gray[h - kk:, w - kk:].mean()
    if region == "四角":
        return [tl, tr, bl, br]
    if region == "上部":
        return [tl, tr]
    # 边缘带
    top = gray[:kk, :].mean()
    bot = gray[h - kk:, :].mean()
    if h > 2 * kk:
        left = gray[kk:h - kk, :kk].mean()
        right = gray[kk:h - kk, w - kk:].mean()
    else:
        left = gray[:, :kk].mean()
        right = gray[:, w - kk:].mean()
    return [top, bot, left, right]


def _luma(img):
    """img: [H, W, 3] → [H, W] 亮度，系数与量测脚本一致。"""
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])


def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


def _robust(vals):
    """剔掉最离群的一个采样块，剩下取平均。

    为什么不直接取中位数：四个角的绝对亮度差得很远（实测 72 / 140 / 39 / 49），
    「比值的中位数」和「均值的比值」不是一回事 —— 中位数会让暗角占更大权重，
    算出的增益偏大（黑蕾丝那版从 +0.26 恶化到 +2.17）。
    为什么不直接取平均：及地长裙会伸进下两角，把背景采样变成衣服采样，
    平均会被拉高，真正的背景角就欠校正 5~7 个灰阶。

    折中：先按中位数找出最离群的那块丢掉，剩下的取平均 ——
    既保住均值的无偏性，又能扛住 1 个角被人物侵占。
    """
    if len(vals) <= 2:
        return sum(vals) / len(vals) if vals else 0.0
    med = _median(vals)
    if len(vals) == 3:
        return med
    worst = max(range(len(vals)), key=lambda i: abs(vals[i] - med))
    rest = [v for i, v in enumerate(vals) if i != worst]
    return sum(rest) / len(rest)


def _smooth(vals, win):
    """一维移动平均（边界用可用窗口），压掉逐帧估计的抖动。"""
    n = len(vals)
    if win <= 1 or n < 3:
        return vals
    half = max(1, win // 2)
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(vals[lo:hi]) / (hi - lo))
    return out


class FxAiToneMatch:
    CATEGORY = "凤希AI/视频"
    FUNCTION = "run"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图片序列",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片序列": ("IMAGE", {
                    "tooltip": "生成出来的帧序列（解码、裁剪之后）。"}),
                "方式": (["增益", "偏移", "不启用"], {
                    "default": "增益",
                    "tooltip": "增益：整帧乘一个系数。黑位基本不动（P1 5.4→6.1），\n"
                               "高光恢复得多，最适合「发闷」这种观感。默认。\n"
                               "偏移：整帧加一个固定灰阶差。对比度绝对不变，\n"
                               "但会把黑蕾丝/黑皮裤一起抬亮（P1 5.4→14.9）而发灰。\n"
                               "不启用：原样透传，不采任何样、不改任何像素。\n"
                               "        排查画面问题时先用它排除本节点的嫌疑。"}),
                "采样区": (["八方位", "四角", "上部", "边缘带", "全图"], {
                    "default": "八方位",
                    "tooltip": "拿哪块区域来估计「背景亮度」。\n"
                               "八方位：3×3 网格去掉中心，8 块一起估。默认，最稳。\n"
                               "四角：4 个角块。能扛住 1~2 个角被人物侵占。\n"
                               "上部：只有上两角。及地长裙/阔腿裤挡满下半幅时用这个。\n"
                               "边缘带：上下左右四条边带。适合人物占满左右的镜头。\n"
                               "全图：只在空镜用，有人物时会被服装带偏。\n"
                               "背景全黑（舞台、夜景）时四角/八方位都会采到噪声，\n"
                               "算出的比值会被下面的范围检查拦掉，需要换「上部」再试。"}),
                "强度": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "校正量乘这个系数。填 0 等于关闭本节点。"}),
                "最大调整": ("FLOAT", {
                    "default": 20.0, "min": 0.0, "max": 40.0, "step": 0.5,
                    "tooltip": "单位是灰阶（0-255）。校正量超过这个值就按上限截断，\n"
                               "防止参考段是闪白/黑场时把整段带偏。"}),
                "逐帧": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "关：整段用同一个校正量（实测暗化不漂移，推荐关）。\n"
                               "开：逐帧各自估计，再按「平滑帧数」做移动平均。\n"
                               "开销差别很小：216 帧实测 0.28s vs 0.39s。"}),
                "平滑帧数": ("INT", {
                    "default": 15, "min": 1, "max": 121, "step": 2,
                    "tooltip": "只在「逐帧」开启时生效。窗口越大越稳、越不容易闪。"}),
            },
            "optional": {
                "参考序列": ("IMAGE", {
                    "tooltip": "源片的对应帧（同一段、逐帧对齐）。\n"
                               "接上就自动标定；不接就原样透传。"}),
            },
        }

    def run(self, 图片序列, 方式="增益", 采样区="八方位", 强度=1.0,
            最大调整=20.0, 逐帧=False, 平滑帧数=15, 参考序列=None):
        if 方式 == "不启用":
            print("[凤希AI·影调] 未启用，原样透传")
            return (图片序列,)
        if 图片序列 is None or 图片序列.shape[0] == 0:
            return (图片序列,)
        if 参考序列 is None or 参考序列.shape[0] == 0:
            print("[凤希AI·影调] 没接参考序列，原样透传")
            return (图片序列,)

        s = float(强度 or 0.0)
        limit = float(最大调整 or 0.0) / 255.0
        if s <= 0.0 or limit <= 0.0:
            return (图片序列,)

        cur = 图片序列.to(torch.float32)
        ref = 参考序列.to(torch.float32)
        nb, h, w = cur.shape[0], cur.shape[1], cur.shape[2]
        k = max(16, min(h, w) // 8)

        def patches(t, i):
            return _patch_means(_luma(t[min(i, t.shape[0] - 1)]), 采样区, k)

        cur_p = [patches(cur, i) for i in range(nb)]
        ref_p = [patches(ref, i) for i in range(nb)]
        mid = nb // 2

        if 方式 == "偏移":
            per_frame = [_robust([r - c for r, c in zip(rp, cp)])
                         for rp, cp in zip(ref_p, cur_p)]
            est = per_frame if 逐帧 else [_median(per_frame)] * nb
            est = [max(-limit, min(limit, d)) for d in est]
            if 逐帧:
                est = _smooth(est, int(平滑帧数))
            else:
                est = [d * s for d in est]
            out = cur.clone()
            if 逐帧:
                for i in range(nb):
                    out[i] = cur[i] + est[i] * s
            else:
                out = cur + est[0]
            print("[凤希AI·影调] 生成 %s → 源 %s ｜ 各块差 %s ｜ 采用 %+.2f 灰阶"
                  "（%s，强度 %.2f）"
                  % ("/".join("%.0f" % (v * 255) for v in cur_p[mid]),
                     "/".join("%.0f" % (v * 255) for v in ref_p[mid]),
                     "/".join("%+.1f" % ((r - c) * 255)
                              for r, c in zip(ref_p[mid], cur_p[mid])),
                     est[mid] if 逐帧 else est[0], 采样区, s))
        else:
            per_frame = []
            for rp, cp in zip(ref_p, cur_p):
                ratios = [(r / c) if c > 1e-4 else 1.0 for r, c in zip(rp, cp)]
                per_frame.append(_robust(ratios))
            est = per_frame if 逐帧 else [_median(per_frame)] * nb

            # 采样给出的结论本身不可信时直接跳过 —— 这一段画面不动，
            # 交给下游的「接缝色彩对齐」按上一段末帧平移，段间不会跳
            base = est[mid] if 逐帧 else est[0]
            if not (GAIN_MIN <= base <= GAIN_MAX):
                print("[凤希AI·影调] 生成 %s → 源 %s ｜ 各块比 %s ｜ 比值 ×%.4f 超出 "
                      "%.1f~%.1f，判定采样不可靠，本段跳过（%s）"
                      % ("/".join("%.0f" % (v * 255) for v in cur_p[mid]),
                         "/".join("%.0f" % (v * 255) for v in ref_p[mid]),
                         "/".join("%.3f" % ((r / c) if c > 1e-4 else 1.0)
                                  for r, c in zip(ref_p[mid], cur_p[mid])),
                         base, GAIN_MIN, GAIN_MAX, 采样区))
                return (图片序列,)

            cur_m = _robust(cur_p[mid])
            max_g = 1.0 + limit / max(cur_m, 1e-3)
            est = [max(1.0 / max_g, min(max_g, g)) for g in est]
            if 逐帧:
                est = _smooth(est, int(平滑帧数))
            if 逐帧:
                out = cur.clone()
                for i in range(nb):
                    out[i] = cur[i] * (1.0 + (est[i] - 1.0) * s)
                g = 1.0 + (est[mid] - 1.0) * s
            else:
                g = 1.0 + (est[0] - 1.0) * s
                out = cur * g
            print("[凤希AI·影调] 生成 %s → 源 %s ｜ 各块比 %s ｜ 采用 ×%.4f"
                  "（%s，强度 %.2f）"
                  % ("/".join("%.0f" % (v * 255) for v in cur_p[mid]),
                     "/".join("%.0f" % (v * 255) for v in ref_p[mid]),
                     "/".join("%.3f" % ((r / c) if c > 1e-4 else 1.0)
                              for r, c in zip(ref_p[mid], cur_p[mid])),
                     g, 采样区, s))

        out = out.clamp_(0.0, 1.0).to(图片序列.dtype)
        return (out,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
