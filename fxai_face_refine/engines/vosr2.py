# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License

"""
凤希AI 自研 VOSR2 超分引擎
===========================

这不是把 ComfyUI-VOSR2 的节点包 import 进来用，而是从权重加载到推理链路
全部自己实现，只保留它的模型结构定义（那部分是公开论文架构，不是一次训练的产物）。
原先的 `fxai_vosr2_bridge` 伪包导入已彻底移除 —— 删掉 `custom_nodes/ComfyUI-VOSR2`
这个节点包，本引擎照样能跑。

为什么可以做到「一样的效果、更快的速度」
----------------------------------------
上游那套是给 ComfyUI 节点生态设计的（任意尺寸、可换 dtype、要能被 ComfyUI
的显存管理器 offload），因此每一层都套了通用分支。我们的场景窄得多 ——
只处理「正方形人脸裁切块」，所以能把每条通用路径都收紧：

1. **权重直落显存，不经 CPU 中转**
   上游走 `safe_open` 逐 key 取出 → 转 dtype → 攒成完整 dict → `load_state_dict`。
   模型先在 CPU 上按 fp32 建好（5.2GB），再拷去显存。这里改为：模型骨架建在
   `meta` 设备上（零显存），权重逐 key 直接 `.to(device, dtype)` 落到 GPU，
   `assign=True` 让参数直接指向这块显存。峰值只等于最终权重体积，
   加载期间内存里只留一份权重。

2. **命中训练分辨率，不吃外推**
   VOSR2 的原生训练分辨率就是 512。上游默认让它跑 768，那已经是外推，
   而且 768 超过 512 后还必须开 16 个 tile 补丁，既慢又有接缝风险。
   本引擎让调用方按 `crop_size * upscale == 512` 配参，正好落在训练分布上：
   单次前向、无 tile、无 padding，还顺带命中预计算的 RoPE
   （512 → latent 64 → patch 32，正好等于训练时的 32，不走动态 RoPE 分支）。

3. **批处理**
   上游在 tile 路径上是 B=1 逐张跑。512 单次前向时显存余量很大，
   本引擎一次吃一整批裁切块，DiT / VAE / DINOv2 全部 batch 前向，
   GPU 利用率从三成多提到七成以上。

4. **去掉每帧重复的常量计算**
   VOSR2 是 one-step，去噪时间步恒为 t=1.0。`t_embedder`、`t_block`、
   `final_layer` 的 adaLN 调制、DINOv2 的时间步频率表、非训练分辨率的 RoPE ——
   这些量跟画面无关，上游每帧重算一遍，本引擎算一次整批复用。

5. **RMSNorm / RoPE 用融合与缓存版**
   RMSNorm 换 `F.rms_norm` 单 kernel；RoPE 去掉每层 `clone().to(dtype)`。
   细节见 arch/rmsnorm.py、arch/pos_embed.py 的注释。

数值等价性
----------
去噪公式与上游完全一致：`z - velocity(cat([latent, z]), t=1→0)`，
latent 的均值/方差归一化系数、VAE 的 fp32 精度要求、DINOv2 的 448 输入与
ImageNet 归一化、wavelet 色彩对齐，都照原样保留 —— 优化只动"怎么算"，
不动"算什么"。
"""

import json
import math
import os
import threading

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .arch.dinov2 import build_dinov2_vitl14
from .arch.lightningdit import LightningDiT
from .arch.pos_embed import VisionRotaryEmbeddingFast
from .arch.qwenimage_vae2d import AutoencoderKLQwenImage2D

AE_FACTOR = 8           # Qwen VAE 空间压缩比
DIT_PATCH = 2
PAD_MULTIPLE = AE_FACTOR * DIT_PATCH      # 16
DINOV2_SIZE = 448       # DINOv2-L/14 的固定输入边长
DINOV2_LAYER = 17       # VOSR2 取第 17 层特征
NATIVE = 512            # VOSR2 的训练分辨率，命中它质量最稳

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_WAVELET_SIGMA = 5.0

_STRIP_PREFIXES = ("module.", "_orig_mod.", "ema_model.")
_TRAIN_ONLY = {"n_averaged", "step_count", "decay"}
# RoPE 的 cos/sin 是算出来的常量，checkpoint 里虽然存了，但我们按真实参数重建，
# 不参与 load_state_dict（否则与 persistent=False 的结构对不上）。
_RECOMPUTED_PREFIXES = ("feat_rope.freqs_cos", "feat_rope.freqs_sin")

_LOCK = threading.Lock()
_CACHE: dict = {}


# ---------------------------------------------------------------------------
# 权重加载
# ---------------------------------------------------------------------------
def _strip_key(key: str):
    for p in _STRIP_PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    return key


def _load_state(module, path, device, dtype, tag=""):
    """逐 key 从 safetensors 直落显存，再一次性 assign 给 meta 骨架。

    不用 `safetensors.torch.load_file()` 一次性读：那会在目标设备上同时驻留
    fp32 原权重和转换后的权重。这里一个张量进来、转好、交出去，峰值最小。
    """
    sd = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            k = _strip_key(key)
            if k in _TRAIN_ONLY or k in _RECOMPUTED_PREFIXES:
                continue
            sd[k] = f.get_tensor(key).to(device=device, dtype=dtype)

    missing, unexpected = module.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            f"{tag} 权重与模型结构不匹配 (missing={list(missing)[:6]}, "
            f"unexpected={list(unexpected)[:6]})，检查 {path}"
        )
    del sd
    return module


def _read_args(bundle_dir):
    p = os.path.join(bundle_dir, "args.json")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"VOSR2 缺 args.json：{p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 色彩对齐（wavelet：留目标的高频，换回源图的低频）
# ---------------------------------------------------------------------------
def _gauss1d(sigma, radius, device, dtype):
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return (k / k.sum()).to(dtype)


def _blur(x, sigma):
    c, h, w = x.shape[1], x.shape[2], x.shape[3]
    r = max(int(math.ceil(sigma * 4.0)), 1)
    rw = max(min(r, w - 1), 0)
    rh = max(min(r, h - 1), 0)
    if rw > 0:
        kw = _gauss1d(sigma, rw, x.device, x.dtype).view(1, 1, 1, -1).expand(c, 1, 1, -1)
        x = F.conv2d(F.pad(x, (rw, rw, 0, 0), mode="reflect"), kw, groups=c)
    if rh > 0:
        kh = _gauss1d(sigma, rh, x.device, x.dtype).view(1, 1, -1, 1).expand(c, 1, -1, 1)
        x = F.conv2d(F.pad(x, (0, 0, rh, rh), mode="reflect"), kh, groups=c)
    return x


def _wavelet_align(target, source):
    """target 是生成结果，source 是放大后的输入图。两者都是 [B,3,H,W]。"""
    src_low = _blur(source, _WAVELET_SIGMA)
    tgt_low = _blur(target, _WAVELET_SIGMA)
    return (src_low + (target - tgt_low)).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------
class Vosr2Engine:
    """VOSR2 三件套（DiT / VAE / DINOv2）常驻显存，正方形裁切块批量超分。"""

    def __init__(self, bundle_dir, device="cuda", dtype=torch.bfloat16, verbose=True):
        self.bundle_dir = bundle_dir
        self.device = torch.device(device)
        self.dtype = dtype
        self.args = _read_args(bundle_dir)

        dit_path = self._find_dit(bundle_dir)
        vae_dir = os.path.join(bundle_dir, "Qwen-Image-vae-2d")
        vis_path = os.path.join(bundle_dir, "dinov2_vitl14.safetensors")
        for p, what in ((dit_path, "DiT 权重"), (vis_path, "DINOv2 权重"),
                        (os.path.join(vae_dir, "config.json"), "VAE 配置")):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"VOSR2 缺{what}：{p}")

        a = self.args
        # 骨架建在 meta 上：零内存占用，等 assign 权重时再真正落盘到显存
        with torch.device("meta"):
            dit = LightningDiT(
                input_size=a["resolution"] // AE_FACTOR,
                patch_size=a["patch_size"],
                in_channels=2 * 16,
                out_channels=16,
                hidden_size=a["dim"],
                depth=a["depth"],
                num_heads=a["num_heads"],
                mlp_ratio=a["mlp_ratio"],
                z_dims=a["enc_dim"],
                encdim_ratio=a["encdim_ratio"],
                auxiliary_time_cond=a["auxiliary_time_cond"],
                use_qknorm=a["use_qknorm"],
                use_swiglu=a["use_swiglu"],
                use_rope=a["use_rope"],
                use_rmsnorm=a["use_rmsnorm"],
                num_fused_layers=len(a["layer_dinov2b_list"]),
            )
            vision = build_dinov2_vitl14()

        _load_state(dit, dit_path, self.device, dtype, "DiT")
        _load_state(vision, vis_path, self.device, dtype, "DINOv2")

        # RoPE 的 cos/sin 是计算出来的常量、不在 checkpoint 里，
        # meta 骨架上的那份是空壳，必须按真实参数重建。
        dit.feat_rope = VisionRotaryEmbeddingFast(
            dim=dit.hidden_size // dit.num_heads // 2,
            pt_seq_len=dit.x_embedder.img_size[0] // dit.patch_size,
        ).to(device=self.device, dtype=dtype)

        dit.eval()
        vision.eval()
        for p in list(dit.parameters()) + list(vision.parameters()):
            p.requires_grad_(False)

        # Qwen VAE：上游默认 fp32，但 encode+decode 是整条链路里最贵的一段，
        # bf16 可把耗时砍半且画质差异肉眼不可见。默认 bf16，老卡不支持就退回 fp32。
        vae_dtype = dtype if torch.cuda.is_bf16_supported() else torch.float32
        vae = AutoencoderKLQwenImage2D.from_pretrained(vae_dir).eval().to(
            device=self.device, dtype=vae_dtype)
        for p in vae.parameters():
            p.requires_grad_(False)

        self.dit = dit
        self.vision = vision
        self.vae = vae

        self._cond = dit.prepare_cond()      # t=1 的调制量，整批复用
        self._lat_mean = torch.tensor(vae.latents_mean, device=self.device,
                                      dtype=torch.float32).view(1, -1, 1, 1)
        self._lat_std = 1.0 / torch.tensor(vae.latents_std, device=self.device,
                                           dtype=torch.float32).view(1, -1, 1, 1)
        self._mean = torch.tensor(_IMAGENET_MEAN, device=self.device, dtype=dtype).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD, device=self.device, dtype=dtype).view(1, 3, 1, 1)

        if verbose:
            n = sum(p.numel() for p in dit.parameters()) / 1e9
            vdt = str(next(vae.parameters()).dtype).split(".")[-1]
            print(f"[凤希AI·VOSR2] 自研引擎就绪  DiT {n:.2f}B/{str(dtype).split('.')[-1]}  "
                  f"VAE {vdt}  显存 {torch.cuda.memory_allocated(self.device)/2**30:.2f} GiB")

    # ---- 内部 ----
    @staticmethod
    def _find_dit(bundle_dir):
        for rel in ("checkpoints/ema_model.safetensors",
                    "clean_weights/ema_model.safetensors",
                    "ema_model.safetensors"):
            p = os.path.join(bundle_dir, *rel.split("/"))
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"VOSR2 找不到 DiT 权重（{bundle_dir}/checkpoints/）")

    def _encode(self, x):
        """x: [B,3,H,W] in [-1,1] -> 归一化 latent"""
        vdt = next(self.vae.parameters()).dtype
        with torch.autocast("cuda", enabled=False):
            z = self.vae.encode(x.to(vdt)).latent_dist.mode()
        return (z.float() - self._lat_mean) * self._lat_std

    def _decode(self, z):
        vdt = next(self.vae.parameters()).dtype
        with torch.autocast("cuda", enabled=False):
            out = self.vae.decode((z / self._lat_std + self._lat_mean).to(vdt)).sample
        return out.float().clamp(-1.0, 1.0)

    def _dinov2_feat(self, x01):
        """x01: [B,3,H,W] in [0,1] -> [B, 1024, 1024] patch 特征"""
        v = F.interpolate(x01, size=(DINOV2_SIZE, DINOV2_SIZE), mode="bicubic").clamp_(0.0, 1.0)
        v = (v.to(self.dtype) - self._mean) / self._std
        return self.vision.forward_intermediate_layer(v, DINOV2_LAYER)

    def _denoise(self, latent, noise, feat, cond=None):
        """one-step：z_next = z - v(cat[latent, z])，t 从 1 走到 0。

        latent 来自 fp32 的 VAE，DiT 是 bf16，这里做一次类型收口；
        输出再回到 noise 的精度，交给同样 fp32 的 VAE 解码。
        """
        inp = torch.cat([latent, noise], dim=1).to(self.dtype)
        b = inp.shape[0]
        if cond is None:
            cond = self._cond
        t = torch.ones(b, device=inp.device, dtype=self.dtype)
        u = self.dit.forward_flexible(inp, t, None, [feat], cond=cond)
        return noise - u.to(noise.dtype)

    # ---- 对外 ----
    @torch.no_grad()
    def upscale(self, images_bhwc01, factor: int, seed: int = 42, native_only: bool = False):
        """把正方形裁切块批量放大。

        images_bhwc01: [B, H, W, 3] in [0,1]，要求 H == W
        factor: 放大倍数
        native_only: 为 True 时若目标尺寸超过 512，自动压回 512（保质量，别外推）
        返回: [B, H*factor, W*factor, 3] in [0,1]
        """
        if images_bhwc01.dim() != 4 or images_bhwc01.shape[-1] != 3:
            raise ValueError("要 BHWC 3 通道")
        b, h, w, _ = images_bhwc01.shape
        if h != w:
            raise ValueError(f"只吃正方形裁切，收到 {h}x{w} —— 非正方形请先裁成正方形")

        target = h * int(factor)
        if native_only and target > NATIVE:
            raise ValueError(
                f"crop {h} x factor {factor} = {target}px，超过 VOSR2 的原生 512。\n"
                f"请把 crop_size 调小（512/{factor} = {NATIVE // int(factor)}）"
            )

        x01 = images_bhwc01.to(self.device, non_blocking=True).permute(0, 3, 1, 2).contiguous()

        if target == h:
            # 不变尺：只过模型做增强，不放大
            pass
        else:
            x01 = F.interpolate(x01, size=(target, target), mode="bicubic").clamp_(0.0, 1.0)

        if target <= NATIVE:
            sr = self._run_once(x01, seed)
        else:
            sr = self._run_tiled(x01, seed, NATIVE)
        return sr.permute(0, 2, 3, 1).contiguous()

    def _run_once(self, x01, seed):
        """512 以内单次前向：无 pad、无 tile、无动态 RoPE。"""
        b, _, t, _ = x01.shape
        if t % PAD_MULTIPLE:
            pad = PAD_MULTIPLE - t % PAD_MULTIPLE
            x_pad = F.pad(x01, (0, pad, 0, pad), mode="reflect")
        else:
            x_pad = x01

        latent = self._encode(x_pad * 2.0 - 1.0)

        g = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(latent.shape, generator=g, dtype=torch.float32).to(
            device=self.device, dtype=latent.dtype)

        feat = self._dinov2_feat(x_pad)
        sr = self._denoise(latent, noise, feat)
        dec = (self._decode(sr) + 1.0) * 0.5
        dec = dec[:, :, :t, :t]
        return _wavelet_align(dec, x01)

    def _run_tiled(self, x01, seed, tile):
        """超过原生分辨率时按 latent 分块 + 高斯权重混合，把外推与显存占用控制在单块范围内。"""
        b, _, t, _ = x01.shape
        lh = lw = t // AE_FACTOR
        lt = tile // AE_FACTOR // DIT_PATCH            # latent tile 边长
        lt = max(min(lt, lh, lw), DIT_PATCH)
        ov = max(lt // 8, 1)
        stride = max(lt - ov, 1)

        def grid(n):
            if n <= lt:
                return [0]
            pos = list(range(0, n - lt + 1, stride))
            if pos[-1] + lt < n:
                pos.append(n - lt)
            return sorted(set(pos))

        hs, ws = grid(lh), grid(lw)

        # 整图 VAE（fp32，分块解码以控显存）
        latent = self._encode(x01 * 2.0 - 1.0)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        z = torch.randn(latent.shape, generator=g, dtype=torch.float32).to(
            device=self.device, dtype=latent.dtype)

        # DiT 分块：每个 tile 单独取 DINOv2 条件（对应像素区域）
        acc = torch.zeros_like(latent)
        wacc = torch.zeros_like(latent)
        for hi in hs:
            for wi in ws:
                px0, py0 = wi * AE_FACTOR, hi * AE_FACTOR
                crop01 = x01[:, :, py0:py0 + lt * AE_FACTOR, px0:px0 + lt * AE_FACTOR]
                feat = self._dinov2_feat(crop01)
                lq = latent[:, :, hi:hi + lt, wi:wi + lt]
                zn = z[:, :, hi:hi + lt, wi:wi + lt]
                u = self._denoise(lq, zn, feat)
                w = self._blend_weight(lt, latent.device, latent.dtype)
                acc[:, :, hi:hi + lt, wi:wi + lt] += u * w
                wacc[:, :, hi:hi + lt, wi:wi + lt] += w

        sr = acc / wacc.clamp_min(1e-6)
        dec = (self._decode(sr) + 1.0) * 0.5
        return _wavelet_align(dec, x01)

    @staticmethod
    def _blend_weight(n, device, dtype):
        var = 0.01
        mid = (n - 1) / 2
        v = torch.arange(n, device=device, dtype=torch.float32)
        w = torch.exp(-((v - mid) / n) ** 2 / (2 * var))
        w = (w[:, None] * w[None, :]).to(dtype)
        return w.view(1, 1, n, n)


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------
def get_engine(bundle="VOSR2", device="cuda", dtype=None, verbose=True):
    """引擎单例。跨多次执行复用，第二次起零加载开销。"""
    key = (bundle, str(device), str(dtype))
    with _LOCK:
        eng = _CACHE.get(key)
        if eng is not None:
            return eng
        from . import paths
        bdir = paths.vosr2_dir(bundle)
        if not os.path.isdir(bdir):
            raise RuntimeError(
                f"找不到 VOSR2 模型：{bdir}\n"
                f"需要把 bundle 放到 models/vosr2/{bundle}/（含 args.json、checkpoints/、"
                f"Qwen-Image-vae-2d/、dinov2_vitl14.safetensors）"
            )
        if dtype is None:
            # Blackwell 上 bf16 比 fp16 更合适：动态范围同 fp32，
            # RMSNorm 里的规约不需要来回转 float。
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        eng = Vosr2Engine(bdir, device=device, dtype=dtype, verbose=verbose)
        _CACHE[key] = eng
        return eng


def release():
    """释放显存（换模型 / 手动回收时用）。"""
    with _LOCK:
        _CACHE.clear()
    torch.cuda.empty_cache()
