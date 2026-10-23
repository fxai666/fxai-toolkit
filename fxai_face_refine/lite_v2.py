# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权

"""
凤希AI - 人脸精修 · 轻量链 V2（全脸）
====================================

画面里所有模糊的小脸都修，不只修一个人。

修哪张脸
--------
判据只有一条：**脸宽 < 处理脸宽上限（默认 96px）的全修。**
参考图不参与这个判断，它只决定「哪张脸算哪个角色」，也就是换不换脸。

四条硬约束
----------
1. `裁切框边长 × 超分倍数` 落在 512（VOSR2 训练分辨率）
2. 裁切框按**这张脸自己的宽度**算，与全片最大脸无关（小脸占比才够）
3. 脸宽 ≥ 处理脸宽上限就跳过，整张不动（唯一的速度总闸）
4. 阈值附近带 15% 滞回：同一张脸在相邻帧之间保持同一判定，不跟着检测抖动进出

默认行为
--------
- 换脸：参考图里匹配上的角色各换各的，`换脸脸宽上限` 之内才换
- 认谁：每张脸自动跟所有参考图比一次，最像谁就归谁，各换各的
- 默认人物：`默认人物` 填 N，整帧一张脸又认不出时当成第 N 张写回（默认第 1 张，0 = 不写回）

`lite.py` 是公共工具层（裁切 / 超分 / 贴回 / 换脸），节点类只在本文件。
"""

import time

import cv2
import numpy as np
import torch

from comfy.utils import ProgressBar

from .engines import detect as fx_detect
from .engines import vosr2 as fx_vosr
from .lite import (
    _apply_swap,
    _as_image_batch,
    _as_image_list,
    _auto_crop,
    _bgr2t,
    _crop_square,
    _ellipse_mask,
    _paste,
    _resolve_device,
    _snap_crop,
    _superres_batch,
    _t2bgr,
    _bc,
    _bc_done,
)

_INTERP = cv2.INTER_LANCZOS4

_EVT_START = "face_refine_v2_start"
_EVT_DONE = "face_refine_v2_done"

# 脸宽判定滞回带宽（相对 处理脸宽上限）。
# 脸宽正好卡在阈值附近时，检测框会一帧大一点、一帧小一点，裸判断就是
# 「修一帧、跳一帧、再修一帧」，成片上看就是脸在闪。V1 靠身份跟踪做了 15%
# 滞回，V1 下线后这段逻辑丢了，这里用「上一帧同一张脸框」的 IoU 关联补回来：
#   已经在修的，要胖到 lo×1.15 才停；
#   已经跳过的，要瘦到 lo×0.85 才重新开始修。
_HYST = 0.15
_HYST_IOU = 0.30      # 低于这个 IoU 就当成另一张脸，重新裸判断


def _iou(a, b) -> float:
    """两个 x1,y1,x2,y2 框的交并比。"""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class FxAiFaceRestoreLiteV2:
    """全脸版：画面里所有模糊的小脸都修，不只修一个人。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "输入帧": ("IMAGE",),
                "处理脸宽上限": ("INT", {
                    "default": 96, "min": 0, "max": 4096, "step": 8,
                    "tooltip": "**唯一的总闸**：脸宽 ≥ 这个值就跳过，不裁、不超分、不贴回，"
                               "原样输出。默认 96 —— 本来清楚的脸不去动它，省掉大部分算力。\n"
                               "0 = 全片每张脸都处理。\n"
                               "V2 是「所有小于这个值的脸都修」，不是只修一张。\n"
                               "阈值附近有 15% 滞回：已经在修的胖到 ×1.15 才停，"
                               "已经跳过的瘦到 ×0.85 才开修，防止脸在阈值上下抖时闪帧。"}),
                "裁切框边长": ("INT", {
                    "default": 0, "min": 0, "max": 512, "step": 16,
                    "tooltip": "裁切框边长（原图尺度）。**0 = 自动**：每张脸按自己的宽度算，"
                               "取 1.5 倍脸宽并向上对齐到 128/256/512，超分倍数自动配套，"
                               "保证 裁切框 × 超分倍数 正好 512（VOSR2 训练分辨率）。\n"
                               "手填时所有脸共用一个框，只允许 128/256/512。"}),
                "超分倍数": ("INT", {
                    "default": 4, "min": 1, "max": 4, "step": 1,
                    "tooltip": "超分倍数。有效放大倍数 = 512 / 裁切框边长。"
                               "裁切框边长 = 0（自动）时由程序配套设定，这一项不生效。"}),
                "超分引擎": (["自研超分 (VOSR2)", "纯插值 (Lanczos)"], {
                    "default": "自研超分 (VOSR2)",
                    "tooltip": "自研超分 (VOSR2) = 生成式超分，能把糊掉的五官重建出来；\n"
                               "纯插值 (Lanczos) = 只放大不救细节（没装模型时用）。"}),
                "遮罩比例": ("FLOAT", {
                    "default": 0.70, "min": 0.40, "max": 1.20, "step": 0.05,
                    "tooltip": "椭圆 mask 相对脸宽的比例。脸大就调小，贴回区域收到脸部。"}),
                "羽化半径": ("FLOAT", {
                    "default": 0.045, "min": 0.0, "max": 0.20, "step": 0.005,
                    "tooltip": "羽化半径（相对裁切框）。调大接缝更软，但会削弱修脸力度。"}),
                "身份相似度阈值": ("FLOAT", {
                    "default": 0.10, "min": 0.02, "max": 0.80, "step": 0.02,
                    "tooltip": "**认人判定线**：每帧最像参考图的那张脸，"
                               "相似度 ≥ 这个值才写回。\n"
                               "**它不决定修不修** —— 修不修只看 处理脸宽上限。\n"
                               "远景小脸（35~50px）与参考图的相似度常在 0.02~0.20，默认 0.10。\n"
                               "漏写回往下调 0.06 / 0.04；认错人往上调 0.20 / 0.30。\n"
                               "**认不出时**：整帧只有一张脸、又没过线，"
                               "按 默认人物 那张写回。\n"
                               "没接参考图时这一项不生效。"}),
                "默认人物": ("INT", {
                    "default": 1, "min": 0, "max": 15, "step": 1,
                    "tooltip": "**认不出是谁的时候，当成第几张参考图里那个人。**\n"
                               "正常流程是每张脸跟所有参考图比一次、最像谁就归谁，用不到这一项。\n"
                               "只有「整帧一张脸、又一张都没过判定线」时才拿它写回。\n"
                               "1 = 第 1 张（默认），2 = 第 2 张，以此类推。\n"
                               "0 = 不写回，认不出就只超分。\n"
                               "多人同框不启用 —— 认不出谁是谁的时候贴上去就是贴错人。\n"
                               "参考图里有多张脸时，取面积最大的那张。"}),
                "换脸脸宽上限": ("INT", {
                    "default": 96, "min": 0, "max": 4096, "step": 2,
                    "tooltip": "**换脸的分界线**：脸宽小于它才换，大于它只超分。默认 96。\n"
                               "⚠️ 必须 ≤ 处理脸宽上限，否则自动按处理上限算。\n"
                               "判据：46px 脸 纯超分 +12.7% / 换脸 **+15.1%**，换脸占优；\n"
                               "121px 脸 纯超分 **+31.6%** / 换脸 −27.1%，换脸毁图。\n"
                               "脸 ≤85px 走 128 裁切框、放大 4 倍，超分信息不足；\n"
                               "脸 >85px 走 256 裁切框、放大 2 倍，超分自身够用。\n"
                               "0 = 不判断，脸多大都换（不推荐）。"}),
                "换脸强度": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "换脸结果的合成强度。1.0 = 完全用换脸结果；"
                               "0.7 左右 = 与超分结果混合，身份弱一点但更贴原画面。"}),
                "肤色回拉": ("FLOAT", {
                    "default": 0.60, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "把换脸后的肤色拉回原脸。inswapper 的肤色跟着目标图走，"
                               "暗光/暖光镜头不拉会变成一张发白的面具。"}),
                "纹理保留": ("FLOAT", {
                    "default": 0.90, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "把超分救回来的高频（皮肤质感、发丝）叠回换脸结果。默认 0.9。\n"
                               "inswapper 输出只有 128 空间的信息量，不叠会明显偏糊。\n"
                               "判据（60px 脸）：纹理 0.5 比纯超分低 7.5%，0.9 高 50%；\n"
                               "80px：0.5 低 12.4%，0.9 高 33.8%。"}),
                "随机种子": ("INT", {"default": 42, "min": 0, "max": 2 ** 31 - 1}),
                "批大小": ("INT", {
                    "default": 4, "min": 1, "max": 32, "step": 1,
                    "tooltip": "一次送进超分引擎的裁切块数量。4 最快；batch 8 会因显存挤占掉速。"}),
                "显卡编号": ("INT", {"default": 0, "min": 0, "max": 7,
                    "tooltip": "用第几张显卡。\n"
                               "⚠️ ComfyUI 在 Windows 上默认把 CUDA_VISIBLE_DEVICES 锁成 0，"
                               "填 1 会自动改用 0 号卡并给出提示。"
                               "真想用第二张，启动时加 --cuda-device all。"}),
            },
            "optional": {
                "参考图": ("IMAGE", {
                    "tooltip": "**认人用**：接一张 = 一个角色；接多张 = 多个角色，各换各的。\n"
                               "**不影响修不修** —— 所有 <处理脸宽上限 的脸都会超分；\n"
                               "它决定「这张脸算哪个角色」，也就是换成谁的脸。\n"
                               "不接也能正常跑：全部只超分，不换脸。",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("处理后帧", "处理报告")
    FUNCTION = "run"
    CATEGORY = "凤希AI/人脸精修"
    DESCRIPTION = (
        "V2 全脸版：画面里**所有模糊的小脸都修**，不只修一个人。\n"
        "【修不修】只看 处理脸宽上限：脸宽 < 96px 的全修，更大的原样输出。\n"
        "【换脸】默认生效：参考图里匹配上的角色，超分后各自换成自己的脸；"
        "没匹配上的只超分，绝不会被换成别人的脸。\n"
        "【参考图】接上才换脸；不接也能跑（全部只超分）。\n"
        "【认谁】每张脸自动跟所有参考图比一次，最像谁就归谁，各换各的，没有主角。\n"
        "【默认人物】整帧一张脸、又认不出是谁时，当成第几张参考图那位写回（默认第 1 张，0 = 不写回）。\n"
        "【裁切】默认按每张脸自己的宽度自动定框，保证 裁切×倍数=512。\n"
        "【防闪】阈值附近 15% 滞回，同一张脸不会一帧修一帧不修。"
    )

    def run(self, 输入帧=None, 处理脸宽上限=96, 裁切框边长=0, 超分倍数=4,
            超分引擎="自研超分 (VOSR2)", 遮罩比例=0.70, 羽化半径=0.045,
            身份相似度阈值=0.10, 默认人物=1,
            换脸脸宽上限=96, 换脸强度=1.0, 肤色回拉=0.60, 纹理保留=0.90,
            随机种子=42, 批大小=4, 显卡编号=0, 参考图=None, **legacy):
        # 兼容旧的英文参数名（老脚本直接调 run() 时不会因为改名而挂）
        if 输入帧 is None:
            输入帧 = legacy.get("images")
        # 旧名兼容：强制参考图序号 是 -1 关 / 0 起数，默认人物 要 +1（-1 → 0 不写回，0 → 1 第 1 张）
        if "强制参考图序号" in legacy:
            默认人物 = int(legacy["强制参考图序号"]) + 1
        elif "主角是第几张参考图" in legacy:
            默认人物 = int(legacy["主角是第几张参考图"])
        elif "兜底用第几张参考图" in legacy:
            默认人物 = int(legacy["兜底用第几张参考图"])
        # 「换成第几张参考图 / 换脸目标图序号」已下线：把所有人换成同一个人没有意义。
        # 老工作流带着这两个名字进来，静默忽略，不报错也不改行为。
        默认人物 = max(0, int(默认人物))
        images = _as_image_batch(输入帧)
        ref_frames = _as_image_list(参考图)
        engine = "lanczos" if "插值" in str(超分引擎) or "lanczos" in str(超分引擎).lower() else "vosr2"

        n = int(images.shape[0])
        if n == 0:
            return (images, "没有输入帧")

        device_id, dev_warn = _resolve_device(显卡编号)
        notes = [dev_warn] if dev_warn else []
        dev = f"cuda:{device_id}"

        # 前面节点（视频采样模型之类）可能还占着显存。启动参数带 --disable-smart-memory 时
        # ComfyUI 不会自动卸载它们，本节点又要把 VOSR2（1.39B）+ inswapper + buffalo_l
        # 一起装进来（约 6.5 GiB），撞上就 OOM —— 而且 OOM 时不会崩，是**静默降级**：
        # 超分全部退回插值、换脸直接不初始化，成片糊一大截。
        # 所以加载自己的模型之前先腾一次地方；只有真的不够才动手，免得显存充裕时白卸别人的模型。
        try:
            import comfy.model_management as _mm
            _free, _total = torch.cuda.mem_get_info(device_id)
            _need = 6800 * 1024 ** 2          # 本节点实测峰值约 6.2 GiB，留 0.6 GiB 余量
            if _free < _need:
                _freed = _mm.free_memory(_need - _free, torch.device(dev))
                _mm.soft_empty_cache()
                _free2, _ = torch.cuda.mem_get_info(device_id)
                notes.append(
                    f"显存紧张（空闲 {_free / 1024 ** 3:.1f} GiB < 需要 "
                    f"{_need / 1024 ** 3:.1f} GiB），已卸载 {len(_freed)} 个其他模型，"
                    f"现空闲 {_free2 / 1024 ** 3:.1f} GiB")
        except Exception as exc:
            notes.append(f"⚠️ 显存预清理跳过（不影响运行）：{exc}")

        t0 = time.time()
        h, w = int(images.shape[1]), int(images.shape[2])
        msg = (f"人脸精修V2(全脸) 开始执行：{n} 帧 {w}x{h}  "
               f"参考图 {len(ref_frames)} 张  引擎 {engine}")
        print(f"[凤希AI] {msg}", flush=True)
        _bc(_EVT_START, {
            "message": msg, "阶段": "开始", "版本": "V2全脸",
            "帧数": n, "宽": w, "高": h, "参考图张数": len(ref_frames),
            "引擎": engine, "处理脸宽上限": 处理脸宽上限,
            "显卡编号": device_id, "批大小": 批大小,
        })

        det = fx_detect.get_detector(device_id=device_id)
        sr_engine = fx_vosr.get_engine(device=dev) if engine == "vosr2" else None

        # ---- 1) 参考图身份：既做归属统计，也做默认人物 ----
        refs = []      # [(名字, 向量)]
        for k, rf in enumerate(ref_frames):
            rb = _t2bgr(rf)
            f = det.detect(rb, order=0)
            if f is None or f.embedding is None:
                notes.append(f"⚠️ 参考图 #{k} 没检测到人脸，已跳过")
                continue
            refs.append((f"参考#{k}", f.embedding))

        # 角色表：refs 里每个 (名字, 向量) 就是一个角色。正常路径是每张脸跟所有
        # 参考图比一次、最像谁就归谁，没有「主角」这个概念。
        # 默认人物 只在整帧一张都没过判定线时用：拿指定的那张参考图写回。
        # 参数从 1 数起，取索引减 1，和用户数图的方式对齐；0 = 不写回。
        fb_k = -1
        want_fb = int(默认人物)
        if want_fb > 0 and refs:
            fb_k = next((j for j, (nm, _e) in enumerate(refs)
                         if nm == f"参考#{want_fb - 1}"), -1)
            if fb_k < 0:
                notes.append(f"⚠️ 默认人物={want_fb} 不可用"
                             f"（没这张或这张没检出人脸），认不出时不写回")
        gate = float(身份相似度阈值)
        if refs:
            notes.append(
                f"认人：{len(refs)} 个角色，判定线 {身份相似度阈值:.2f}"
                + (f"；认不出时按 参考#{want_fb - 1} 写回" if fb_k >= 0 else "；认不出不写回"))

        # ---- 换脸（默认开）：匹配上参考图的每张脸都换，匹配不上的只超分 ----
        swapper = None
        swap_latents = {}       # 角色下标 -> 换脸身份 latent
        swap_fail = 0
        swap_skip = 0
        swap_strength = float(换脸强度)
        skin_pull = float(肤色回拉)
        texture_keep = float(纹理保留)
        swap_max_face = int(换脸脸宽上限)
        if not refs:
            notes.append("没接参考图，认不出角色，本次只做超分")
        else:
            if (swap_max_face > 0 and 处理脸宽上限 > 0
                    and swap_max_face > 处理脸宽上限):
                notes.append(f"⚠️ 换脸脸宽上限 {swap_max_face} > 处理脸宽上限 "
                             f"{处理脸宽上限}，实际按 {处理脸宽上限} 生效")
                swap_max_face = 处理脸宽上限
            try:
                from .engines import swap as fx_swap
                swapper = fx_swap.get_swapper(device_id=device_id)
            except Exception as exc:
                notes.append(f"⚠️ 换脸初始化失败，本次只做超分：{exc}")
                swapper = None
            if swapper is not None:
                # 每个角色换自己：把糊脸换成同一个人的清晰版本。
                # 画面里是谁还是谁，绝不会被换成别人。
                for k, (nm, _e) in enumerate(refs):
                    ref_i = int(nm.split("#")[1])
                    tf = det.detect(_t2bgr(ref_frames[ref_i]), order=0)
                    if tf is None or tf.embedding is None:
                        notes.append(f"⚠️ 换脸素材 #{k} 没检出人脸，该角色只超分")
                        continue
                    swap_latents[k] = swapper.encode_source(tf.embedding)
                if swap_latents:
                    notes.append(
                        f"换脸：{len(swap_latents)} 个角色各换各的，"
                        f"强度 {swap_strength:.2f} / 肤色回拉 {skin_pull:.2f} / "
                        f"纹理保留 {texture_keep:.2f}")
                else:
                    swapper = None

        # ---- 2) 逐帧检测：所有脸都要，只看够不够小 ----
        lo = float(处理脸宽上限) if 处理脸宽上限 > 0 else 0.0
        pbar = ProgressBar(n)
        frames_bgr = []
        tasks = []          # [(帧号, Face, 角色下标)]   角色下标 -1 = 只超分不换脸
        skip_clear = 0
        hyst_hold = 0       # 被滞回保住、没跟着阈值抖的脸数
        owner = {}          # 归属统计：{(参考名): 张数}
        hero_hit = 0        # 相似度过确认线的张数
        solo_fallback = 0   # 认不出写回：整帧一张脸、没过线但按默认人物写回的帧数
        prev_boxes = []     # 上一帧的 [(bbox, 上一帧有没有修)]，给滞回做关联
        owner["其他/未归属"] = 0
        for name, _e in refs:
            owner[name] = 0
        for i in range(n):
            bgr = _t2bgr(images[i])
            frames_bgr.append(bgr)
            faces = [f for f in det.detect_all(bgr) if f.w >= 16]

            # 每个角色在这一帧**最多认一张脸**（取最像的）。不加这个约束，
            # 同一个人在一帧里出现两次（镜面反射、误检）会被换两遍。
            assign = {}          # 角色下标 -> (相似度, Face)
            solo_here = set()    # 本帧按默认人物写回的脸（id），归属统计里算给默认人物那张
            for f in faces:
                if f.embedding is None or not refs:
                    continue
                # 这张脸整张不动（≥ 处理脸宽上限），角色名额留给会被处理的小脸。
                # 不挡的话，同帧一张清晰大脸会把名额占掉，小脸反而拿不到角色。
                if lo > 0 and f.w >= lo:
                    continue
                sims = [float(np.dot(f.embedding, e)) for _n2, e in refs]
                k = int(np.argmax(sims))
                if sims[k] < gate:
                    continue
                if k not in assign or sims[k] > assign[k][0]:
                    assign[k] = (sims[k], f)

            # 默认人物：整帧只有一张脸、又一张都没过判定线时，当成 默认人物 那位写回。
            # 多人帧不写回 —— 认不出谁是谁的时候贴上去就是贴错人。
            if (not assign and fb_k >= 0 and len(faces) == 1
                    and faces[0].embedding is not None
                    and (lo <= 0 or faces[0].w < lo)):
                sims = [float(np.dot(faces[0].embedding, e)) for _n2, e in refs]
                assign[fb_k] = (sims[fb_k], faces[0])
                solo_here.add(id(faces[0]))
                solo_fallback += 1

            role_of = {id(hf): k for k, (_s, hf) in assign.items()}

            cur_boxes = []
            for f in faces:
                # 归属统计（不影响修不修）
                if refs and f.embedding is not None:
                    sims = [float(np.dot(f.embedding, e)) for _n2, e in refs]
                    best = int(np.argmax(sims))
                    if sims[best] >= 身份相似度阈值:
                        hero_hit += 1
                    if id(f) in solo_here:           # 按默认人物写回，算给默认人物那张
                        owner[refs[fb_k][0]] += 1
                    elif sims[best] >= 身份相似度阈值:
                        owner[refs[best][0]] += 1
                    else:
                        owner["其他/未归属"] += 1
                # 修不修：主判据是脸宽，卡在阈值附近时带上滞回
                if 处理脸宽上限 <= 0:
                    do_it = True
                else:
                    # 找上一帧位置最重叠的那张脸，看它当时修没修
                    last_flag, best = None, _HYST_IOU
                    for pb, pflag in prev_boxes:
                        v = _iou(pb, f.bbox)
                        if v > best:
                            best, last_flag = v, pflag
                    if last_flag is None:
                        do_it = f.w < lo                      # 新出现 / 刚跳回画面
                    elif last_flag:
                        do_it = f.w < lo * (1.0 + _HYST)      # 正在修：放宽才停
                        if do_it and f.w >= lo:
                            hyst_hold += 1
                    else:
                        do_it = f.w < lo * (1.0 - _HYST)      # 已跳过：更小才开修
                        if not do_it and f.w < lo:
                            hyst_hold += 1
                cur_boxes.append((f.bbox, do_it))
                if not do_it:
                    skip_clear += 1
                    continue
                tasks.append((i, f, role_of.get(id(f), -1)))
            prev_boxes = cur_boxes
            pbar.update(1)

        if not tasks:
            rep = notes + [
                f"帧数 {n}   没有需要修的脸",
                (f"清晰跳过 {skip_clear} 张脸（脸 ≥ {处理脸宽上限}px）"
                 if skip_clear else "全片没检出可用的人脸"),
                "已原样返回。想让清楚的脸也过一遍，把 处理脸宽上限 调大或设 0。",
            ]
            res = (images, "\n".join(rep))
            _bc_done(res, t0)
            return res

        # ---- 3) 按「这张脸自己的宽度」定裁切框，同尺寸的归一组 ----
        if 裁切框边长 > 0:
            cs, up = _snap_crop(裁切框边长)
            if cs != 裁切框边长:
                notes.append(f"⚠️ 裁切框 {裁切框边长} 不是 VOSR2 原生尺寸，已吸附到 {cs}")
            groups = {(cs, up): tasks}
            crop_desc = f"{cs}x{up}（全部脸共用）"
        else:
            groups = {}
            for t in tasks:
                cs, up = _auto_crop(t[1].w)
                groups.setdefault((cs, up), []).append(t)
            crop_desc = " / ".join(
                f"{c}x{u}（{len(g)} 张）"
                for (c, u), g in sorted(groups.items()))

        notes.insert(0, (f"帧数 {n}   裁切 {crop_desc} = 512px"
                         + ("  （命中 VOSR2 原生 512）" if all(c * u == fx_vosr.NATIVE
                                                             for c, u in groups) else "")))

        # ---- 4) 分批超分 + 逐张羽化贴回 ----
        outs = [b.copy() for b in frames_bgr]
        sr_fail = 0
        panel = ProgressBar(len(tasks))
        for (gcrop, gup), gtasks in groups.items():
            for start in range(0, len(gtasks), 批大小):
                chunk = gtasks[start:start + 批大小]
                crops, boxes, fboxes = [], [], []
                for (i, f, _role) in chunk:
                    c, box = _crop_square(frames_bgr[i], f, gcrop)
                    crops.append(c)
                    boxes.append(box)
                    fboxes.append([f.bbox[0] - box[0], f.bbox[1] - box[1],
                                   f.bbox[2] - box[0], f.bbox[3] - box[1]])
                try:
                    bigs = _superres_batch(sr_engine, crops, gup, 随机种子)
                except Exception as exc:
                    sr_fail += len(crops)
                    print(f"⚠️[凤希AI] 超分失败，本批退回插值: {exc}")
                    bigs = _superres_batch(None, crops, gup, 随机种子)

                for j, (i, f, role) in enumerate(chunk):
                    big, crop = bigs[j], crops[j]
                    small = cv2.resize(big, (crop.shape[1], crop.shape[0]),
                                       interpolation=cv2.INTER_AREA)
                    mask = _ellipse_mask(crop.shape[0], fboxes[j], 遮罩比例, 羽化半径)
                    # 同一帧可能有多张脸，累加贴回
                    outs[i] = _paste(outs[i], small, boxes[j], mask, 1.0)
                    # 这张脸匹配上了参考图里的某个角色 -> 再叠一层换脸。
                    # 没匹配上的（role < 0）只保留超分结果，绝不会被换成别人的脸。
                    if (swapper is not None and role >= 0
                            and role in swap_latents and f.kps is not None):
                        if swap_max_face > 0 and f.w >= swap_max_face:
                            swap_skip += 1
                        else:
                            try:
                                outs[i] = _apply_swap(
                                    outs[i], frames_bgr[i], f, swapper,
                                    swap_latents[role], 遮罩比例, 羽化半径,
                                    swap_strength, skin_pull, texture_keep)
                            except Exception as exc:
                                if swap_fail == 0:      # 只打第一条
                                    print(f"⚠️[凤希AI] 换脸失败，本张退回超分结果: {exc}")
                                swap_fail += 1
                    panel.update(1)

        # ---- 5) 汇总 ----
        n_swap = sum(1 for (_i, _f, r) in tasks if r >= 0)
        rep = notes + [
            f"链路：裁 {crop_desc} -> "
            f"{'自研 VOSR2' if sr_engine else 'lanczos'} -> 逐张羽化贴回",
            f"共修 {len(tasks)} 张脸（{n} 帧，平均 {len(tasks) / n:.2f} 张/帧）",
        ]
        if skip_clear:
            rep.append(f"清晰跳过 {skip_clear} 张脸（脸 ≥ {处理脸宽上限}px，整张不动，省算力）")
        if hyst_hold:
            rep.append(f"滞回保持 {hyst_hold} 张（脸宽在 {处理脸宽上限}px 附近抖动，"
                       f"按上一帧状态走，不跟着闪）")
        if refs:
            rep.append("参考图归属：")
            for name, _e in refs:
                rep.append(f"  {name}：{owner[name]} 张")
            if owner["其他/未归属"]:
                rep.append(f"  其他/未归属：{owner['其他/未归属']} 张")
            rep.append(f"  其中相似度过 {身份相似度阈值:.2f} 的 {hero_hit} 张")
            if solo_fallback:
                rep.append(f"认不出写回 {solo_fallback} 帧"
                           f"（整帧一张脸、没过 {身份相似度阈值:.2f} 线，"
                           f"按默认人物 {refs[fb_k][0]} 写回）")
        if swapper is not None:
            rep.append(f"换脸：{n_swap} 张脸匹配上参考图并换脸，其余只超分")
            if swap_skip:
                rep.append(f"换脸跳过 {swap_skip} 张（脸 ≥ {swap_max_face}px，"
                           f"这个尺寸超分自己就够，换脸反而更糊）")
        if swap_fail:
            rep.append(f"⚠️ 有 {swap_fail} 张换脸失败，已退回纯超分结果")
        if sr_fail:
            rep.append(f"⚠️ 有 {sr_fail} 张脸超分失败，已退回插值 —— "
                       f"这部分脸会比预期糊。显存不够时会出现，先腾显存再跑")
        rep.append(f"算力：检测 {det.device} / 超分 "
                   f"{f'GPU(bf16) {sr_engine.device}' if sr_engine else 'CPU'}")
        if swapper is not None:
            tail = "；匹配上参考图的脸再换脸，其余保持本人"
        elif refs:
            tail = "；换脸没起来，本次只超分（原因见上方 ⚠️）"
        else:
            tail = "；没接参考图，未换脸"
        rep.append("✅ 所有模糊小脸都超分" + tail)

        out_t = torch.stack([_bgr2t(o) for o in outs], dim=0)
        res = (out_t, "\n".join(rep))
        _bc_done(res, t0, evt=_EVT_DONE)
        return res
