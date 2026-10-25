# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权

"""SeedVR2 一步超分 —— 单节点封装版。

官方蓝图要连 5 个节点（PreProcess / VAEEncodeTiled / Conditioning / KSampler /
PostProcessing），还得自己算 4n+1 补帧、自己切块、自己拼回来。这里一个节点做完：

    缩放 → 4n+1 对齐 → VAE 编码 → 一步采样 → VAE 解码 → 色彩回拉 → 裁回原帧数

分块按**像素帧**切，不按潜帧：一块进去，编码 / 采样 / 解码 / 校色全在这块里跑完，
写回输出后立刻释放。所以内存和显存只跟「一块多大」有关，跟整段多少帧无关。
显存不够时自动把块减半重试，不会整段跑一半 OOM 崩掉。
"""

import math
import os
import time

import torch

import comfy.model_management as mm
import comfy.sample
import comfy.sd
import comfy.utils
import folder_paths
from comfy.ldm.seedvr.color_fix import (
    adain_color_transfer,
    lab_color_transfer,
    wavelet_color_transfer,
)
from comfy.utils import ProgressBar

VAE_DOWNSAMPLE = 8          # BYTEDANCE_VAE_SPATIAL_DOWNSAMPLE
VAE_TEMPORAL = 4            # 4n+1 里的那个 4
DEFAULT_VAE_KEYWORD = "seedvr"
MIN_CHUNK = 5               # OOM 降块的下限：再小就没有时间上下文了

# 自动分块帧数的取值范围（4n+1）。上限 41 有实测背书：
# 3.6 Mpx 下 41 帧/块峰值显存 9.36 GiB，只比 21 帧的 9.11 GiB 高 0.25 GiB。
AUTO_CHUNK_MIN = 21
AUTO_CHUNK_MAX = 41

# 「分块帧数」下拉框的选项：自动 + 常用 4n+1 值。
# 保留整数选项是为了兼容老工作流里存的 21 / 41 这类数值。
CHUNK_CHOICES = ["自动", 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 61, 81, 121]

# ---- 耗时模型（六组实测拟合，误差 ±3%）----
#
# 关键发现：**编码和解码吃的东西不一样。**
#
#   编码 ∝ 帧次 × 分块像素    ← 受 VAE 分块影响，重叠区要重算
#   解码 ∝ 帧次 × 输出像素    ← 跟分块完全无关
#
# 六组实测反推（帧次 = 块数 × chunk，见 plan_units）：
#
#   组                 编码/帧次/Mpx   解码/帧次/输出Mpx
#   A 3B 块512  c21        0.552            0.957
#   B 3B 块512  c37        0.517            0.932
#   E 3B 块768  c41        0.420            0.954
#   G 3B 736×1152         0.403            0.954
#   H 3B 1280×768         0.398            0.980
#   F 7B 块1024           0.396            0.965
#
# 解码那一列稳得可怕 —— 换模型、换分块、换分辨率全在 0.93~0.98。
# 所以**调分块只能动编码那 30~40%，解码那 60% 一动不动**。
# 这就是「二维分块算下来省 10%，实跑只快 4.5%」的原因。
#
# 编码系数随「每帧块数」略涨：块越多，每块的固定开销（建 mask、搬运、
# 边界羽化）摊得越散。512 要切 15 块，所以是 0.52~0.55；768 只切 6 块，0.42。
ENC_PER_MPX = 0.395         # 每帧次每百万分块像素的基础秒数
ENC_PER_TILE = 0.0085       # 每多一个块，再加这么多
DEC_PER_MPX = 0.95          # 每帧次每百万**输出**像素的秒数
EXTRA_RATIO = 1.09          # 缩放 + 采样 + 校色 ≈ 编解码的 9%

# safetensors 8G 出头，每次重跑都从盘上读一遍太慢，缓存一份
_MODEL_CACHE = {}
_VAE_CACHE = {}

_COLOR_FN = {
    "lab": lab_color_transfer,
    "wavelet": wavelet_color_transfer,
    "adain": adain_color_transfer,
}

_OOM_ERRORS = tuple({torch.cuda.OutOfMemoryError, mm.OOM_EXCEPTION})


# ==========================================
# 基础工具
# ==========================================

def _available_ram():
    """当前可用物理内存（字节）。拿不到就返回 None，预检自动跳过。"""
    try:
        import psutil
        return psutil.virtual_memory().available
    except Exception:
        return None


def plan_blocks(total, chunk, overlap):
    """按分块规则算实际会切几块。"""
    if chunk <= 0:
        return 0
    stride = max(1, int(chunk) - int(overlap))
    return max(1, math.ceil((int(total) - int(overlap)) / float(stride)))


def plan_units(total, chunk, overlap):
    """总共要算多少「帧次」。

    每块都会补满到 chunk 帧（4n+1 补齐），所以是 `块数 × chunk`，
    不是 `总帧数 × 浪费比`。

    这个区别在块少的时候很致命：72 帧 / 41 帧块 = 2 块，
    真实是 2×41 = 82 帧次，而 `72 × 41/(41-8) = 89` 多算了 9%。
    """
    blocks = plan_blocks(total, chunk, overlap)
    return blocks * int(chunk) if blocks else 0


def estimate_seconds(total, th, tw, chunk, overlap, tile_xy=None):
    """预估耗时。

    帧次 × (编码单价 × 分块像素 + 解码单价 × 输出像素) × 附加开销。

    三块都得算对，缺一块就会偏 20% 以上：

    * **帧次** = `块数 × chunk`，每块都补满（见 plan_units）。
      不能用 `总帧数 × 浪费比` —— 那假设块无限多，2 块时多算 9%
    * **分块像素**：所有块加起来、末块按边界裁过（见 tile_plan），
      重叠区重复算，块越小重复越多
    * **输出像素**：解码只认这个，跟分块无关（见上面那张六组实测表）

    `tile_xy` 传 None 时按「不切块」算（只在前面的流程还没定块大小时用）。
    """
    if chunk <= 0:
        return 0.0
    units = plan_units(total, chunk, overlap)
    out_mpx = (th * tw) / 1e6
    if tile_xy:
        n_tile, calc_px = tile_plan(tw, th, tile_xy[0], tile_xy[1])
        tiled_mpx = calc_px / 1e6
    else:
        n_tile, tiled_mpx = 1, out_mpx
    k_enc = ENC_PER_MPX + ENC_PER_TILE * n_tile
    return units * (k_enc * tiled_mpx + DEC_PER_MPX * out_mpx) * EXTRA_RATIO


def align_4n1(n):
    """把帧数补到 4n+1，跟官方 cut_videos 的规则完全一致。"""
    if n <= 1:
        return n
    if n <= 4:
        return VAE_TEMPORAL + 1
    return n + (VAE_TEMPORAL - ((n - 1) % VAE_TEMPORAL)) % VAE_TEMPORAL


def snap_4n1(n):
    """把用户填的帧数往下取到合法的 4n+1。"""
    n = int(n)
    if n <= 1:
        return 1
    if n < VAE_TEMPORAL + 1:
        return VAE_TEMPORAL + 1
    return VAE_TEMPORAL * ((n - 1) // VAE_TEMPORAL) + 1


def _want_chunk(v):
    """把「分块帧数」的输入归一成 int：0 = 自动，>0 = 用户指定的帧数。

    下拉框给的是 int，但老工作流可能存的是字符串，两种都认。
    """
    if isinstance(v, str):
        s = v.strip()
        if not s or s == "自动":
            return 0
        try:
            return int(float(s))
        except ValueError:
            return 0
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def pick_chunk(total, overlap, want=0):
    """挑「分块帧数」。want > 0 就按用户填的吸附到 4n+1；want = 0 自动。

    自动的逻辑：在 21~41 的合法值里，找**总算力单元数最小**的那个。

        块数 = ceil((总帧 - 重叠) / (块大小 - 重叠))
        单元 = 块数 × 块大小          ← 重叠的那几帧每块都要重算一遍

    这跟「取最接近总帧数的 4n+1」不完全一样，后者会吃亏 —— 60 帧时最接近的是 29，
    但 29 要 3 块共 87 个单元，37 只要 2 块共 74 个。**按单元数挑才是真的省**。

    块越大重叠重复占比越低，所以趋势是「越大越快」，上限卡在 41：
    实测 41 帧/块在 3.6 Mpx 下峰值显存 9.36 GiB，是安全值。

    段本身不到 21 帧时下限跟着缩小（11 帧用 13、22 帧用 25）——
    22 帧按 21 切会变成 21+9 两块要算 42 个单元，比一块 25 还贵。
    """
    if want and int(want) > 0:
        return snap_4n1(want)

    lo = AUTO_CHUNK_MIN
    if total < lo:
        lo = max(VAE_TEMPORAL + 1, align_4n1(total))
    ov = max(0, min(int(overlap), (lo - 1) // 2))

    best_c, best_u = lo, None
    c = lo
    while c <= AUTO_CHUNK_MAX:
        blocks = max(1, math.ceil((total - ov) / float(max(1, c - ov))))
        units = blocks * c
        if best_u is None or units < best_u:
            best_c, best_u = c, units
        c += VAE_TEMPORAL
    return best_c


def align16(v):
    """对齐到 16 的倍数（VAE 空间下采样 8，预处理补边 16）。"""
    return max(16, int(round(v / 16.0)) * 16)


def target_size(w, h, scale, short_edge):
    """目标宽高。short_edge > 0 时按短边算，覆盖倍数。"""
    if short_edge and short_edge > 0:
        ratio = float(short_edge) / float(min(w, h))
    else:
        ratio = float(scale)
    return align16(w * ratio), align16(h * ratio)


def _fade_weights(n, device, dtype):
    """块间交叉淡化：中间 1/3 走 Hann，两头留平肩。返回上一块的权重（1 → 0）。"""
    ramp = torch.linspace(0.0, 1.0, steps=n, device=device, dtype=dtype)
    ramp = ((ramp - 1.0 / 3.0) / (1.0 / 3.0)).clamp(0.0, 1.0)
    return (0.5 + 0.5 * torch.cos(torch.pi * ramp)).view(-1, 1, 1, 1)


def _resolve_diffusion_model(model):
    inner = getattr(model, "model", None)
    dm = getattr(inner, "diffusion_model", None)
    if dm is None:
        raise RuntimeError(
            "接进来的 MODEL 不是 SeedVR2 结构（没有 model.diffusion_model）。"
            "把「MODEL」这个输入留空，让节点自己按「模型」那一项加载。"
        )
    return dm


def _seedvr2_models():
    """列出 diffusion_models 里的 SeedVR2 权重。"""
    names = []
    try:
        for name in folder_paths.get_filename_list("diffusion_models"):
            if "seedvr" in os.path.basename(name).lower():
                names.append(name)
    except Exception:
        pass
    return names


def _default_seedvr2_model(names):
    for name in names:
        if "7b" in os.path.basename(name).lower():
            return name
    return names[0] if names else ""


def _find_seedvr2_vae():
    try:
        for name in folder_paths.get_filename_list("vae"):
            if DEFAULT_VAE_KEYWORD in os.path.basename(name).lower():
                return name
    except Exception:
        pass
    return ""


def _load_model(name):
    path = folder_paths.get_full_path_or_raise("diffusion_models", name)
    key = (os.path.abspath(path), os.path.getmtime(path))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    _MODEL_CACHE.clear()
    model = comfy.sd.load_diffusion_model(path)
    _MODEL_CACHE[key] = model
    return model


def _load_vae(name):
    if not name:
        raise RuntimeError(
            "没找到 SeedVR2 的 VAE —— models/vae 下应该有一个文件名带 seedvr 的 safetensors。"
        )
    path = folder_paths.get_full_path_or_raise("vae", name)
    key = (os.path.abspath(path), os.path.getmtime(path))
    cached = _VAE_CACHE.get(key)
    if cached is not None:
        return cached
    _VAE_CACHE.clear()
    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    _VAE_CACHE[key] = vae
    return vae


def build_conditioning(model, latent):
    """把 5D latent 做成 SeedVR2 的 positive / negative（跟官方节点一致）。"""
    dm = _resolve_diffusion_model(model)
    cond = latent.movedim(1, -1).contiguous()
    mask = cond.new_ones(cond.shape[:-1] + (1,))
    cond = torch.cat((cond, mask), dim=-1).movedim(-1, 1)
    positive = [[dm.positive_conditioning.unsqueeze(0), {"condition": cond}]]
    negative = [[dm.negative_conditioning.unsqueeze(0), {"condition": cond}]]
    return positive, negative


def _color_transfer(dec, ref, method):
    """把解码结果的色彩拉回缩放后的输入（逐帧做，显存可控）。

    dec / ref 都是 (帧, 高, 宽, 通道) 的像素序列；color_fix 里的三个函数吃
    (B, C, H, W)，所以进出各搬一次维度。
    """
    if method == "none":
        return dec.clamp(0.0, 1.0)
    if dec.shape[1] != ref.shape[1] or dec.shape[2] != ref.shape[2]:
        ref = comfy.utils.common_upscale(
            ref.movedim(-1, 1), dec.shape[2], dec.shape[1], "bicubic", "disabled"
        ).movedim(1, -1)
    fn = _COLOR_FN[method]
    device = mm.vae_device()
    d = dec.mul(2.0).sub(1.0).movedim(-1, 1).to(device=device)   # (帧, 3, 高, 宽)
    r = ref.mul(2.0).sub(1.0).movedim(-1, 1).to(device=device)
    pieces = [fn(d[i:i + 1], r[i:i + 1]) for i in range(d.shape[0])]
    out = torch.cat(pieces, dim=0).to(device=dec.device)
    del d, r, pieces
    out = out.movedim(1, -1)
    return out.add(1.0).div(2.0).clamp(0.0, 1.0)


def _as_frame_sequence(t):
    """把 VAE 解码结果统一成 (帧, 高, 宽, 通道)。

    SeedVR2 的 decode 可能给 (B,T,H,W,C)、也可能给 (B,C,T,H,W)，单帧还会掉维度，
    这里按通道数落在哪一侧判断，不靠猜。
    """
    if t.ndim == 5:
        if t.shape[-1] in (1, 3, 4):
            b, tt, h, w, c = t.shape
            return t.reshape(b * tt, h, w, c)
        if t.shape[1] in (1, 3, 4):
            b, c, tt, h, w = t.shape
            return t.movedim(1, -1).reshape(b * tt, h, w, c)
        raise RuntimeError(f"看不懂 VAE 解码输出的形状：{tuple(t.shape)}")
    if t.ndim == 4:
        if t.shape[-1] in (1, 3, 4):
            return t
        if t.shape[1] in (1, 3, 4):
            return t.movedim(1, -1)
        raise RuntimeError(f"看不懂 VAE 解码输出的形状：{tuple(t.shape)}")
    if t.ndim == 3:
        return t.unsqueeze(0)
    raise RuntimeError(f"看不懂 VAE 解码输出的形状：{tuple(t.shape)}")


# ---- VAE 空间分块 ----
# 原理：把一帧画面切成若干块，每块独立走一遍 VAE 编解码，再按羽化权重拼回去。
# 所以成本 ≈ 所有块加起来的像素数，重叠区会被重复算。
TILE_MIN = 256
TILE_MAX = 1536
TILE_OVERLAP = 128          # 相邻块的重叠边长（两个方向都用这一个值）
# 单块面积上限 —— 这才是显存的真约束（tile 面积按平方吃显存）。
# 1024×1024 ≈ 1.05 Mpx，3B / 7B 都实测过。
TILE_AREA_CAP = 1024 * 1024


def tile_span(total, tile):
    """沿一个方向切，返回 (块数, 实际计算长度合计)。

    **照抄 comfy.utils.tiled_scale_multidim 的循环**：

        positions = range(0, size - overlap, tile - overlap)
        l = min(tile, size - pos)

    两个要点：
    1. 位置从 0 起、步长 `tile - overlap`，且**只在 `pos < size - overlap` 时切** ——
       不是 `pos < size`。1280 宽用 768 的块，只有 1 块，不是 2 块。
    2. 末块边长是 `size - pos`，会被裁到图像边界内，所以总长度
       **不是 `块数 × tile`**，比那个小。1920 宽用 768 的块：768+768+640=2176，
       不是 3×768=2304。

    第 2 点会实实在在影响选块：按 `块数 × 块面积` 算会把大块的高估 5~20%，
    于是错误地偏向小块。
    """
    total = int(total)
    tile = min(int(tile), total)
    if tile <= 0:
        return 0, 0
    if total <= tile:                      # 这个方向一块就够
        return 1, total
    stride = max(1, tile - TILE_OVERLAP)
    n = 0
    acc = 0
    for pos in range(0, total - TILE_OVERLAP, stride):
        acc += min(tile, total - pos)
        n += 1
    return n, acc


def tile_plan(w, h, tile_x, tile_y):
    """切这么一组块，返回 (块数, 实际要算的像素)。"""
    nx, cx = tile_span(w, tile_x)
    ny, cy = tile_span(h, tile_y)
    return nx * ny, cx * cy


def auto_tile(w, h, area_cap=TILE_AREA_CAP):
    """挑 VAE 分块的长宽，返回 (tile_x, tile_y)。

    穷举所有长宽组合，取「实际要算的像素」最少的那个。

    **长宽要各自独立搜，不能强制正方形。** 因为横向和纵向的最优切法本来就不同：

    | 分辨率 | 正方形最优 | 长宽独立最优 | 省 |
    |---|---|---|---|
    | 1280×768 | 768² → 2 块 / 1.08 Mpx | **1280×768 → 1 块 / 0.98** | 9.1% |
    | 1920×1152 | 1024² → 4 块 / 2.62 | 736×1152 → 3 块 / 2.51 | 4.4% |
    | 3840×1080 | 880² → 10 块 / 5.26 | 880×1088 → 5 块 / 4.70 | 10.6% |
    | 1440×1080 | 784² → 4 块 / 1.89 | 784×1088 → 2 块 / 1.69 | 10.6% |
    | 2560×1536 | 944² → 6 块 / 4.69 | 944×832 → 6 块 / 4.69 | 0% |

    规律是「让块正好整除画面」：横屏就横着切少几刀、竖屏就竖着切，
    扁片（3840×1080）收益最大。2560×1536 这种两个方向都除不尽的，正方形就是最优。

    上限只有 `TILE_AREA_CAP`（单块面积）—— 单边不额外设限，
    因为 1440×768 这种扁块和 1024² 一样吃显存，没理由禁。
    """
    w, h = int(w), int(h)
    if w <= 0 or h <= 0:
        return 512, 512
    area_cap = max(TILE_MIN * TILE_MIN, int(area_cap))
    max_x = min(TILE_MAX, w)
    max_y = min(TILE_MAX, h)

    best = None
    tx = TILE_MIN
    while tx <= max_x:
        ty = TILE_MIN
        while ty <= max_y:
            if tx * ty <= area_cap:
                n, px = tile_plan(w, h, tx, ty)
                # 排序键：先比总计算量，同分时取「单块更小」的（省显存），
                # 再同分取块数少的（接缝少、羽化次数少）
                key = (px, tx * ty, n)
                if best is None or key < best[0]:
                    best = (key, tx, ty)
            ty += 16
        tx += 16

    if best is None:                        # 理论上不会到这儿（256² 永远合法）
        return min(512, w), min(512, h)
    return best[1], best[2]


def _vae_args(tile_x, tile_y):
    return dict(tile_x=tile_x, tile_y=tile_y, overlap=TILE_OVERLAP,
                tile_t=64, overlap_t=8)


def _run_block(model, vae, seg, tw, th, color, seed, vae_args):
    """跑一块：缩放 → 补 4n+1 → 编码 → 一步采样 → 解码 → 校色。

    独立成函数是为了 OOM 时局部变量随栈帧一起释放，不会挂在调用方作用域里。
    返回 (结果, 分段耗时)。分段计时是给「到底哪一步慢」用的 —— 见 run() 报告。
    """
    n = seg.shape[0]
    stage = {}

    # 1) 缩放到目标尺寸 —— SeedVR2 只负责重建，不负责放大
    #    注意：这一步在 **CPU** 上做（seg 本来就在内存里，common_upscale 不搬设备）
    _t = time.time()
    seg_up = comfy.utils.common_upscale(
        seg.movedim(-1, 1), tw, th, "lanczos", "disabled"
    ).clamp_(0.0, 1.0)

    # 2) 补帧到 4n+1（复制最后一帧，跟官方 cut_videos 一致）
    pad_n = align_4n1(n)
    if pad_n > n:
        seg_up = torch.cat([seg_up, seg_up[-1:].repeat(pad_n - n, 1, 1, 1)], dim=0)
    ref = seg_up.movedim(1, -1).contiguous()
    stage["缩放"] = time.time() - _t

    with torch.no_grad():
        # 3) VAE 编码 → 4) 一步采样 → 5) VAE 解码（全在 GPU 上，按空间 tile 搬进搬出）
        _t = time.time()
        latent = vae.encode_tiled(ref, **vae_args)
        stage["编码"] = time.time() - _t

        _t = time.time()
        positive, negative = build_conditioning(model, latent)
        noise = comfy.sample.prepare_noise(latent, seed)
        sampled = comfy.sample.sample(
            model, noise, 1, 1.0, "euler", "simple",
            positive, negative, latent,
            denoise=1.0, disable_pbar=True, seed=seed,
        )
        stage["采样"] = time.time() - _t

        _t = time.time()
        dec = vae.decode_tiled(sampled, **vae_args)
        stage["解码"] = time.time() - _t

    dec = _as_frame_sequence(dec)
    if dec.shape[-1] > 3:
        dec = dec[..., :3]
    if dec.shape[0] < n:
        raise RuntimeError(
            f"VAE 解码只回了 {dec.shape[0]} 帧，块里要的是 {n} 帧 —— "
            "多半是「分块帧数」和 VAE 的时间格对不上。"
        )
    dec = dec[:n]
    ref = ref[:n]

    # 6) 色彩回拉（在 GPU 上逐帧做，见 _color_transfer）
    _t = time.time()
    result = _color_transfer(dec, ref, color)
    stage["校色"] = time.time() - _t
    return result, stage


# ==========================================
# 节点
# ==========================================

class FxAiSeedVR2Upscale:
    """SeedVR2 一步超分：一个节点跑完整条链，按块吃显存。"""

    @classmethod
    def INPUT_TYPES(cls):
        models = _seedvr2_models()
        if not models:
            models = ["（models/diffusion_models 下没找到 SeedVR2 权重）"]
        return {
            "required": {
                "输入帧": ("IMAGE", {
                    "tooltip": "要超分的画面。接图像序列（视频帧列表）也行，单张也行。\n"
                               "会先把整段按「分块帧数」切开，一块一块跑完再拼回来。",
                }),
                "模型": (models, {
                    "default": _default_seedvr2_model(models),
                    "tooltip": "SeedVR2 权重，从 models/diffusion_models 里文件名带 seedvr 的里面选。\n"
                               "**7B 全面强于 3B**：更清楚、身份更稳、瞎编更少。\n"
                               "实测（RTX 5060 Ti）：单帧 1280×768 7B 6~7s / 3B 3.7s；"
                               "批量跑两者几乎一样快（瓶颈在 VAE 分块，不在采样）。\n"
                               "所以默认 7B。3B 只在小显存或赶时间时用。",
                }),
                "放大倍数": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 4.0, "step": 0.25,
                    "tooltip": "目标分辨率 = 原分辨率 × 这个倍数（对齐到 16 的倍数）。\n"
                               "**SeedVR2 自己不缩放**：先用 lanczos 把画面放到这个尺寸，"
                               "模型再在这个尺寸上把细节重建回来。\n"
                               "2 倍性价比最高；4 倍靠的是模型「补」，不是真信息。\n"
                               "填了「目标短边」时这一项不生效。\n"
                               "⚠️ **内存和耗时都按倍数的平方涨**（像素数 = 宽×高×倍数²）。\n"
                               "10 秒（240 帧）768p 竖屏按公式算：1.5 倍 ≈ 15 GB 内存 / 25 分钟；"
                               "2 倍 ≈ 25 GB / 45 分钟；3 倍 ≈ 53 GB / 101 分钟。\n"
                               "⚠️ 这里的「内存」是**整条链**（含下游人脸精修），"
                               "超过可用内存的 90% 节点会直接拒绝开跑，不让你跑到一半卡死。",
                }),
                "目标短边": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 16,
                    "tooltip": "按短边定目标尺寸。**0 = 关掉，用放大倍数**。\n"
                               "填 720 / 1080 / 1440 这类值，不管原图多大都拉到这个短边，"
                               "长边按原比例走。\n"
                               "适合「统一输出到 1080p」这种活。\n"
                               "⚠️ 宽高会对齐到 16 的倍数，填 1080 实际出 1088（差 0.7%）。",
                }),
                "分块帧数": (CHUNK_CHOICES, {
                    "default": "自动",
                    "tooltip": "一块多少帧。必须是 **4n+1**（1/5/9/13/17/21…），"
                               "因为 SeedVR2 的 VAE 时间下采样是 4 倍。\n"
                               "**自动（默认）**：按这一段的实际帧数，在 21~41 之间"
                               "挑一个「重叠重算浪费最小」的 4n+1。\n"
                               "例：120 帧 → 37 帧/块 × 4 块（浪费 1.23×）；"
                               "240 帧 → 37 × 8；262 帧 → 41 × 8（1.25×）。"
                               "比固定 21 帧快 **20% 左右**。\n"
                               "**选了具体数字就按那个跑**，不合规的数会往下取整到 4n+1。\n"
                               "**越大越快**：块间重叠的那几帧每块都要重算，块越大占比越低。\n"
                               "实测 2 倍 1408×2560、30 帧：21 帧/块 9.27 s/帧，"
                               "41 帧/块 **7.64 s/帧**（快 18%），峰值显存只从 9.11 涨到 9.36 GiB。\n"
                               "显存紧就选 9 / 13；整段不超过一块时没有拼缝。",
                }),
                "块间重叠帧": ("INT", {
                    "default": 8, "min": 0, "max": 64, "step": 4,
                    "tooltip": "相邻两块重叠多少帧，用 Hann 曲线交叉淡化接缝。\n"
                               "默认 8。0 = 直接硬拼，长镜头可能看到一闪。\n"
                               "最多取「分块帧数」的一半 —— 再多步长会塌到 1 帧，"
                               "等于每帧都重算一整块，白烧算力。\n"
                               "重叠越大接缝越稳，但重复算的帧也越多。",
                }),
                "色彩校正": (["lab", "wavelet", "adain", "none"], {
                    "default": "lab",
                    "tooltip": "把生成结果的色调拉回缩放后的原图，防止偏色。\n"
                               "**lab** = 在 CIELAB 空间搬颜色，最贴原片（默认）；\n"
                               "**wavelet** = 只搬低频颜色，高频细节留超分的；\n"
                               "**adain** = 只对均值方差，最快，但可能整体染色；\n"
                               "**none** = 不校正，只做几何对齐。",
                }),
                "显存自适应": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "打开时，某一块显存不够会自动把块减半重试，"
                               "并记住这个更小的大小继续跑后面的块。\n"
                               "关掉就严格按「分块帧数」跑，显存不够直接报错。\n"
                               "大分辨率（4K）或小显存卡建议开着。",
                }),
                "随机种子": ("INT", {
                    "default": 42, "min": 0, "max": 2 ** 31 - 1,
                    "tooltip": "一步采样的初始噪声种子。**42 没有任何技术含义** ——"
                               "它只是《银河系漫游指南》里的梗，社区当固定默认值用，"
                               "好处是「同参数同结果」，方便 A/B 对比时排除随机因素。\n"
                               "这里种子对结果的影响**很小**：一步采样、"
                               "且模型被输入画面的 latent 强约束，噪声只决定那一次前向的起点。\n"
                               "真要挑刺才去换它；正常跑就留 42。",
                }),
            },
            "optional": {
                "MODEL": ("MODEL", {
                    "tooltip": "接一个已经加载好的 SeedVR2 模型，就不按「模型」那一项去读了。\n"
                               "一般不用接 —— 节点自己会缓存。",
                }),
                "VAE": ("VAE", {
                    "tooltip": "接一个已经加载好的 SeedVR2 VAE。不接就自动找 "
                               "models/vae 下文件名带 seedvr 的那个。",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT")
    RETURN_NAMES = ("处理后帧", "处理报告", "放大倍数")
    FUNCTION = "run"
    CATEGORY = "凤希AI/视频"
    DESCRIPTION = (
        "SeedVR2 一步超分，一个节点跑完整条链。\n"
        "【它干什么】把画面放大到目标尺寸，再用生成式模型把糊掉的细节重建回来。\n"
        "【和「人脸精修」的区别】SeedVR2 是整帧重绘，身份会漂到 0.76~0.96；"
        "人脸精修只换脸框内的像素、框外一个字节不动，身份能稳在 0.82 以上。\n"
        "**推荐串起来用**：先过 SeedVR2 打底，再过人脸精修把身份拉回来。\n"
        "【分块】按像素帧分块，一块内编码/采样/解码/校色全跑完再释放，"
        "内存只跟块有关，跟总帧数无关。显存不够自动减半重试。\n"
        "【4n+1】块内自动补帧对齐 VAE 时间格，输出再裁回原帧数，不多不少。\n"
        "【放大倍数输出】第三个输出是实际达到的倍数，接到「人脸精修」的「放大倍数」输入，"
        "那边两个脸宽门槛会自动 ×N，不用手算。"
    )

    def run(self, 输入帧=None, 模型="", 放大倍数=2.0, 目标短边=0,
            分块帧数="自动", 块间重叠帧=8, 色彩校正="lab", 显存自适应=True,
            随机种子=42, MODEL=None, VAE=None):

        t_start = time.time()

        if 输入帧 is None:
            raise RuntimeError("「输入帧」没接。")
        image = 输入帧
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise RuntimeError(
                f"「输入帧」应该是 4 维（帧, 高, 宽, 通道），拿到 {tuple(image.shape)}。"
            )

        total, in_h, in_w, in_c = image.shape
        if total < 1:
            raise RuntimeError("「输入帧」是空的。")

        rgb = image[..., :3].float()
        alpha = image[..., 3:] if in_c > 3 else None
        tw, th = target_size(in_w, in_h, 放大倍数, 目标短边)

        notes = []

        # ---- 分块参数 ----
        # 分块帧数 = 自动：按这一段的实际帧数挑一个「重叠浪费最小」的 4n+1
        want_chunk = _want_chunk(分块帧数)
        chunk = pick_chunk(total, 块间重叠帧, want_chunk)
        if want_chunk > 0 and want_chunk != chunk:
            notes.append(f"分块帧数 {want_chunk} 不是 4n+1，已按 {chunk} 跑")
        # 重叠最多占一半，否则步长会塌成 1 帧，等于每帧都重算一整块
        overlap_cap = (chunk - 1) // 2
        overlap = max(0, min(int(块间重叠帧), overlap_cap))
        if int(块间重叠帧) != overlap:
            notes.append(f"块间重叠帧 {int(块间重叠帧)} 超出范围，已按 {overlap} 跑")
        if want_chunk <= 0:
            _blk = max(1, math.ceil((total - overlap) / float(max(1, chunk - overlap))))
            notes.append(
                f"分块帧数自动：{total} 帧 → {chunk} 帧/块 × {_blk} 块"
                f"（重叠浪费 {_blk * chunk / float(total):.2f}×）")

        # ---- 开工前先算账：内存和耗时都是可以提前算准的 ----
        # 块大小要在这里先定下来 —— 耗时预估必须知道「实际算多少像素」
        tile_x, tile_y = auto_tile(tw, th)
        vae_args = _vae_args(tile_x, tile_y)
        n_tile, calc_px = tile_plan(tw, th, tile_x, tile_y)
        calc_mpx = calc_px / 1e6

        # 内存 = 帧数 × 像素数 × 通道 × 4 字节，跟分块无关（分块只管显存峰值）
        out_bytes = total * th * tw * (3 if alpha is None else 4) * 4
        in_bytes = total * in_h * in_w * in_c * 4
        # 整条链还要多一份：下游人脸精修会把整段再复制一遍
        chain_bytes = in_bytes + out_bytes * 2
        # 耗时 = 帧次 × 实际计算像素 × 单帧系数
        est_sec = estimate_seconds(total, th, tw, chunk, overlap, (tile_x, tile_y))
        avail = _available_ram()
        preflight = (
            f"{total} 帧 · {in_w}×{in_h} → {tw}×{th}\n"
            f"输出 {out_bytes / 1024 ** 3:.1f} GiB · "
            f"整链（含下游精修）约 {chain_bytes / 1024 ** 3:.1f} GiB 内存"
        )
        if avail:
            preflight += f" · 当前可用 {avail / 1024 ** 3:.0f} GiB"
        preflight += (
            f"\n预计 {est_sec / 60:.0f} 分钟（{chunk} 帧/块 · 重叠 {overlap} 帧）"
            f"\nVAE 分块 {tile_x}×{tile_y}：{n_tile} 块/帧，实际要算 {calc_mpx:.2f} Mpx/帧"
            f"（输出只有 {tw * th / 1e6:.2f} Mpx，重叠浪费 {calc_mpx / (tw * th / 1e6):.2f}×）"
        )
        print(f"[凤希AI][SeedVR2] 预估：{preflight}")

        if avail and chain_bytes > avail * 0.9:
            keep = max(1, int(total * avail * 0.9 / chain_bytes))
            raise RuntimeError(
                "内存不够，先别跑 —— 跑到一半会被系统换页卡死，比现在报错更糟。\n"
                f"本段 {total} 帧、放大到 {tw}×{th}：整链要 {chain_bytes / 1024 ** 3:.1f} GiB，"
                f"当前只剩 {avail / 1024 ** 3:.1f} GiB。\n"
                f"三条路选一条：\n"
                f"  1. 把这一段切到 {keep} 帧以内（约 {keep / 24:.1f} 秒 @24fps）—— 最省事；\n"
                f"  2. 放大倍数降一档；\n"
                f"  3. 关掉下游的人脸精修，只留超分。"
            )

        # ---- 模型 / VAE ----
        vae = VAE if VAE is not None else _load_vae(_find_seedvr2_vae())
        if MODEL is not None:
            model = MODEL
            model_label = "外部 MODEL"
        else:
            if not 模型 or 模型.startswith("（"):
                raise RuntimeError(
                    "没选到 SeedVR2 模型 —— models/diffusion_models 下应该有一个文件名带 seedvr 的 safetensors。"
                )
            model = _load_model(模型)
            model_label = os.path.basename(模型)

        seed = int(随机种子) & 0xFFFFFFFF
        out = torch.empty(
            (total, th, tw, 3 if alpha is None else 4), dtype=torch.float32
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # ---- 分块跑 ----
        pos = 0
        prev_end = 0
        cur_chunk = chunk
        cur_overlap = overlap
        oom_drops = 0
        blocks = 0
        panel = ProgressBar(total)
        peak_gb = 0.0
        acc = {}            # 分段累计耗时

        while pos < total:
            mm.throw_exception_if_processing_interrupted()
            n = min(cur_chunk, total - pos)
            seg = rgb[pos:pos + n]

            try:
                dec, stage = _run_block(model, vae, seg, tw, th, 色彩校正, seed, vae_args)
            except _OOM_ERRORS:
                del seg
                mm.soft_empty_cache()
                if not 显存自适应 or cur_chunk <= MIN_CHUNK:
                    raise RuntimeError(
                        f"显存不够：{tw}×{th} 下跑 {cur_chunk} 帧/块仍然 OOM。"
                        f"把「分块帧数」调小到 9 或 5，或把「放大倍数」降一档。"
                    )
                new_chunk = max(MIN_CHUNK, snap_4n1((cur_chunk - 1) // 2))
                if new_chunk >= cur_chunk:
                    new_chunk = max(MIN_CHUNK, cur_chunk - VAE_TEMPORAL)
                cur_chunk = new_chunk
                cur_overlap = min(cur_overlap, (cur_chunk - 1) // 2)
                oom_drops += 1
                print(f"[凤希AI][SeedVR2] 显存不够，分块降到 {cur_chunk} 帧重试")
                continue

            # 写回输出：第一块直接铺，后面的块与上一块交叉淡化
            if blocks == 0:
                out[pos:pos + n, :, :, :3] = dec
            else:
                ov_end = min(prev_end, pos + n)
                if ov_end > pos:
                    f = ov_end - pos
                    w = _fade_weights(f, dec.device, dec.dtype)
                    out[pos:ov_end, :, :, :3] = (
                        out[pos:ov_end, :, :, :3] * w + dec[:f] * (1.0 - w)
                    )
                    out[ov_end:pos + n, :, :, :3] = dec[f:]
                else:
                    out[pos:pos + n, :, :, :3] = dec

            if alpha is not None:
                a = alpha[pos:pos + n].float().movedim(-1, 1)
                a = comfy.utils.common_upscale(a, tw, th, "lanczos", "disabled")
                out[pos:pos + n, :, :, 3:] = a.clamp_(0.0, 1.0).movedim(1, -1)
                del a

            if torch.cuda.is_available():
                peak_gb = max(peak_gb, torch.cuda.max_memory_allocated() / (1024 ** 3))

            for _k, _v in stage.items():
                acc[_k] = acc.get(_k, 0.0) + _v

            del seg, dec
            blocks += 1
            prev_end = pos + n
            panel.update(n)
            pos = prev_end - cur_overlap if prev_end < total else total

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        elapsed = time.time() - t_start
        per_frame = elapsed / max(1, total)

        report = (
            f"【SeedVR2 超分】{model_label} · 色彩校正 {色彩校正}\n"
            f"尺寸：{in_w}×{in_h} → {tw}×{th}（{tw / float(in_w):.2f}×）\n"
            f"帧数：{total} 帧（块内补到 4n+1，输出裁回原帧数，不多不少）\n"
            f"分块：{chunk} 帧/块，重叠 {overlap} 帧，实际 {blocks} 块"
            f" · VAE 分块 {tile_x}×{tile_y}（{n_tile} 块/帧，算 {calc_mpx:.2f} Mpx/帧）"
        )
        if cur_chunk != chunk:
            report += f"（中途降到 {cur_chunk} 帧/块）"
        report += (
            f"\n耗时：{elapsed:.1f}s（{per_frame * 1000:.0f} ms/帧）"
            f" · 峰值显存 {peak_gb:.2f} GiB"
            f"\n预估 {est_sec:.0f}s → 实际 {elapsed:.0f}s"
            f"（{elapsed / max(1.0, est_sec) * 100:.0f}%）"
            f"\n内存：本段输出 {out_bytes / 1024 ** 3:.1f} GiB"
            f" · 整链（含下游精修）约 {chain_bytes / 1024 ** 3:.1f} GiB"
        )
        if acc:
            _tot = sum(acc.values()) or 1.0
            _parts = [
                f"{_k} {acc[_k]:.0f}s（{acc[_k] / _tot * 100:.0f}%）"
                for _k in ("缩放", "编码", "采样", "解码", "校色") if _k in acc
            ]
            report += "\n分段：" + " / ".join(_parts)
        report += (
            "\n算力：缩放走 CPU（lanczos，占不到 1%）；"
            "编码 / 采样 / 解码 / 校色全在 GPU —— "
            "编解码按空间 tile 搬进显存、算完搬回内存，所以显存只跟一块有关"
        )
        if oom_drops:
            report += f"\n显存自适应触发 {oom_drops} 次降块"
        if notes:
            report += "\n注：" + "；".join(notes)
        print(f"[凤希AI][SeedVR2] {report}")

        mm.soft_empty_cache()
        # 实际达到的倍数（可能因为「目标短边」或 16 对齐跟「放大倍数」略有出入），
        # 直接喂给「人脸精修」的「放大倍数」输入，那边两个脸宽门槛会自动 ×N。
        return (out, report, tw / float(in_w))
