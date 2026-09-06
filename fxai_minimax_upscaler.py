# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权

"""
MiniMax H3 Latent Upscaler - 简化版
- 纯3D卷积架构
- 仅支持倍数放大 (2x/3x/4x)
- 时间分块节省显存
- 像素对齐防止光带
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import gc
import folder_paths
import re
from einops import rearrange

import fxai_task_store

try:
    import comfy.model_management as mm
    HAS_COMFY_MM = True
except ImportError:
    HAS_COMFY_MM = False

# ==========================================
# 注册模型文件夹
# ==========================================
_LATENT_UPSCALE_FOLDER = "latent_upscale_models"
if _LATENT_UPSCALE_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        _LATENT_UPSCALE_FOLDER,
        os.path.join(folder_paths.models_dir, _LATENT_UPSCALE_FOLDER)
    )

VAE_DOWNSAMPLE = 16

# ==========================================
# 归一化参数 (24通道)
# ==========================================
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.563616402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.772995296404114, 1.895593762397661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523
]

def _make_norm_tensors(device, dtype):
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std

# ==========================================
# 网络组件
# ==========================================
def normalization(channels):
    return nn.GroupNorm(32, channels)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=dropout),
            zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h

class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = normalization(channels)
        self.dwconv = nn.Conv3d(channels, channels,
                                kernel_size=(kernel_size, 1, 1),
                                padding=(padding, 0, 0),
                                groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        identity = x
        h = self.norm(x)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return identity + h

# ==========================================
# 3D网络主体
# ==========================================
class LatentResizer3D(nn.Module):
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=512, dropout=0.1, temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            self.in_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(TemporalConv(channels, temporal_kernel))

        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            self.out_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(TemporalConv(channels, temporal_kernel))

        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale=None, target_size=None, enable_chunking=True):
        if target_size is not None:
            size = target_size
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-3:])
        else:
            return x

        if size == x.shape[-3:]:
            return x

        B, C, T, H, W = x.shape

        tk = 0
        for b in self.in_blocks:
            if isinstance(b, TemporalConv):
                tk = b.dwconv.weight.shape[2]
                break

        overlap = tk
        chunk = 32

        if not enable_chunking or T <= chunk:
            return self._forward_seg(x, scale, size)

        x_padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode='replicate')

        out_full = torch.zeros(B, C, T, size[-2], size[-1], device=x.device, dtype=x.dtype)
        weight_full = torch.zeros(1, 1, T, 1, 1, device=x.device, dtype=x.dtype)

        start = 0
        while start < T:
            seg_start = start
            seg_end = min(T, start + chunk)
            out_start = max(0, seg_start - overlap)
            out_end = min(T, seg_end + overlap)
            lo = max(0, out_start - overlap)
            hi = min(T + 2 * overlap, out_end + overlap)

            seg = x_padded[:, :, lo:hi].contiguous()
            seg_size = (hi - lo, size[-2], size[-1])
            seg_out = self._forward_seg(seg, scale, seg_size)

            s0 = (out_start + overlap) - lo
            s1 = s0 + (out_end - out_start)
            valid_out = seg_out[:, :, s0:s1]
            n_valid = out_end - out_start

            weight = torch.ones(n_valid, device=x.device, dtype=x.dtype)
            if seg_start > out_start:
                blend_len = seg_start - out_start
                weight[:blend_len] = torch.arange(1, blend_len + 1, device=x.device, dtype=x.dtype) / (blend_len + 1)
            if out_end > seg_end:
                blend_len = out_end - seg_end
                weight[-blend_len:] = torch.arange(blend_len, 0, -1, device=x.device, dtype=x.dtype) / (blend_len + 1)

            out_full[:, :, out_start:out_end] += valid_out * weight.view(1, 1, n_valid, 1, 1)
            weight_full[:, :, out_start:out_end] += weight.view(1, 1, n_valid, 1, 1)

            start += chunk
            del seg, seg_out, valid_out

        out_full = out_full / weight_full.clamp(min=1e-8)
        return out_full

    def _forward_seg(self, x, scale, size):
        scale_emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_emb)

        x = self.conv_in(x)
        for b in self.in_blocks:
            if isinstance(b, ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)

        for b in self.out_blocks:
            if isinstance(b, ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x

# ==========================================
# 模型加载
# ==========================================
MODEL_CACHE = {}

def get_models_dir():
    return folder_paths.get_folder_paths(_LATENT_UPSCALE_FOLDER)[0]

def scan_models():
    names = [
        name for name in folder_paths.get_filename_list(_LATENT_UPSCALE_FOLDER)
        if os.path.splitext(name)[1].lower() in (".pth", ".safetensors")
    ]
    return names if names else [f"(请将模型放入: {get_models_dir()})"]

def _load_raw_sd(path):
    if path.endswith('.safetensors'):
        try:
            from safetensors import safe_open
            with safe_open(path, framework="pt", device="cpu") as f:
                sd = {k: f.get_tensor(k) for k in f.keys()}
        except ImportError:
            from safetensors.torch import load_file
            sd = load_file(path, device='cpu')
    else:
        sd = torch.load(path, map_location='cpu', weights_only=False)

    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    sd = {k: v.to(torch.float16) if v.dtype == torch.float8_e4m3fn else v
          for k, v in sd.items()}
    return sd

def _extract_upscaler_sd(sd):
    if any(k.startswith("upscaler.") for k in sd):
        return {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    return sd

def _detect_arch(sd):
    cfg = {
        "in_channels": 24, "in_blocks": 12, "out_blocks": 12, "channels": 512,
        "dropout": 0.1, "temporal_every": 2, "temporal_kernel": 5,
    }
    conv_key = 'conv_in.weight'
    if conv_key in sd:
        cfg["in_channels"] = sd[conv_key].shape[1]
        cfg["channels"] = sd[conv_key].shape[0]

    in_ids, out_ids = set(), set()
    temporal_indices = set()
    for k in sd.keys():
        m = re.match(r'in_blocks\.(\d+)\.in_layers\.', k)
        if m: in_ids.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.in_layers\.', k)
        if m: out_ids.add(int(m.group(1)))
        m = re.match(r'in_blocks\.(\d+)\.dwconv\.weight', k)
        if m: temporal_indices.add(int(m.group(1)))

    if in_ids: cfg["in_blocks"] = len(in_ids)
    if out_ids: cfg["out_blocks"] = len(out_ids)

    if temporal_indices:
        cfg["temporal_every"] = 2
        for k in sd.keys():
            if 'dwconv.weight' in k and k.endswith('dwconv.weight'):
                cfg["temporal_kernel"] = sd[k].shape[2]
                break
    else:
        cfg["temporal_every"] = 0

    return cfg

def load_model(name, device):
    cache_key = f"{name}::{device}"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key].to(device, non_blocking=True)

    try:
        path = folder_paths.get_full_path_or_raise(_LATENT_UPSCALE_FOLDER, name)
    except Exception as e:
        raise FileNotFoundError(f"模型文件未找到: {name}") from e

    raw_sd = _load_raw_sd(path)
    up_sd = _extract_upscaler_sd(raw_sd)
    cfg = _detect_arch(up_sd)

    model = LatentResizer3D(
        in_channels=cfg["in_channels"], in_blocks=cfg["in_blocks"], out_blocks=cfg["out_blocks"],
        channels=cfg["channels"], dropout=cfg["dropout"],
        temporal_every=cfg["temporal_every"], temporal_kernel=cfg["temporal_kernel"],
    )
    model.load_state_dict(up_sd, strict=True)

    # 自动检测模型精度
    sample_sd = up_sd if not up_sd else up_sd
    model_dtype = torch.float32
    for k, v in model.state_dict().items():
        if v.dtype in (torch.float16, torch.bfloat16):
            model_dtype = v.dtype
            break

    model = model.to(device, dtype=model_dtype).eval().requires_grad_(False)

    MODEL_CACHE[cache_key] = model
    return model

# ==========================================
# ComfyUI节点
# ==========================================
class FxAiMiniMaxUpscaler:
    """凤希AI - MiniMax H3 潜空间放大器"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "模型": (scan_models(), {"tooltip": "模型及参考代码由：https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler提供"}),
                "放大倍数": ([2, 3, 4], {"default": 2}),
            },
            "optional": {
                "时间分块": ("BOOLEAN", {"default": True}),
                "释放显存": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "凤希AI/视频"

    def run(self, latent, 模型, 放大倍数, 时间分块=True, 释放显存=True):

        if 模型.startswith('('):
            raise ValueError("请将模型文件放入 latent_upscale_models 目录")

        src = latent["samples"]
        orig_dtype = src.dtype
        was_4d = (src.dim() == 4)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        s = src.to(device=device, copy=True)
        if was_4d:
            s = s.unsqueeze(2)

        b, c, t, h_in, w_in = s.shape
        downsample = VAE_DOWNSAMPLE

        # 计算目标尺寸
        w_out = int(round(w_in * 放大倍数))
        h_out = int(round(h_in * 放大倍数))

        # 像素对齐 (固定32)
        align = 32
        w_pixel = w_out * downsample
        h_pixel = h_out * downsample
        w_pixel_aligned = round(w_pixel / align) * align
        h_pixel_aligned = round(h_pixel / align) * align
        w_out = max(1, int(w_pixel_aligned // downsample))
        h_out = max(1, int(h_pixel_aligned // downsample))

        if w_out == w_in and h_out == h_in:
            return (latent,)

        print(f"[凤希AI] MiniMax放大 - {放大倍数}x")

        try:
            fxai_task_store.broadcast("upscale_start", {
                "放大倍数": 放大倍数,
                "message": f"正在放大：{放大倍数}x"
            })
        except Exception as e:
            print(f"[凤希AI] 放大广播失败：{e}")

        # 加载模型
        model = load_model(模型, device)
        model_dtype = next(model.parameters()).dtype
        norm_mean, norm_std = _make_norm_tensors(device, model_dtype)

        # 将输入转为模型精度
        s = s.to(dtype=model_dtype)

        with torch.inference_mode():
            s_norm = (s - norm_mean) / norm_std
            del s

            out = model(s_norm, scale=放大倍数, target_size=(t, h_out, w_out),
                        enable_chunking=时间分块)

            del s_norm
            out = out * norm_std + norm_mean

        if was_4d:
            out = out.squeeze(2)

        out = out.to(device="cpu", dtype=orig_dtype, non_blocking=True)

        # 释放显存
        if device == "cuda":
            if 释放显存:
                model.to("cpu", non_blocking=True)
            if HAS_COMFY_MM:
                mm.soft_empty_cache()
            else:
                torch.cuda.empty_cache()
            gc.collect()

        try:
            fxai_task_store.broadcast("upscale_done", {
                "放大倍数": 放大倍数,
                "message": f"放大完成：{放大倍数}x"
            })
        except Exception as e:
            print(f"[凤希AI] 放大广播失败：{e}")

        return ({"samples": out},)
