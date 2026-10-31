# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
"""从公开发布的官方权重构建我们自己的模型。

只读权重，不引用任何第三方源码。需要的四个文件（放 `models/FlashVSR/`，
目录名沿用上游品牌，因为权重是他们的）：
  diffusion_pytorch_model_streaming_dmd.safetensors   DiT，825 个张量，fp32
  LQ_proj_in.ckpt                                      LQ 投影，8 个键，bf16
  TCDecoder.ckpt                                       解码器，66 个键，fp32
  posi_prompt.pth                                      固定文本上下文 (1,512,4096)

`Wan2.1_VAE.pth` 是 `full` 模式才要的编码器，我们只用 `tiny-long`，不加载它。
"""

import os

import torch

from .dit import DiTConfig, StreamDiT
from .lq import LQProjector
from .tcdecoder import TCDecoder

DIT_FILE = "diffusion_pytorch_model_streaming_dmd.safetensors"
LQ_FILE = "LQ_proj_in.ckpt"
TCD_FILE = "TCDecoder.ckpt"
CTX_FILE = "posi_prompt.pth"

# 权重目录沿用上游品牌名：模型是他们训练的，目录就归在他们名下
ROOT_NAME = "FlashVSR"


def model_root():
    """定位权重目录 `models/FlashVSR/`。"""
    try:
        import folder_paths
        base = folder_paths.models_dir
    except Exception:
        # 不在 ComfyUI 进程里（独立导入 fxai_vsr）时按包位置向上找：
        # fxai_vsr/ → fxai-toolkit/ → custom_nodes/ → <ComfyUI>/models
        here = os.path.dirname(os.path.abspath(__file__))
        base = os.path.normpath(os.path.join(here, os.pardir, os.pardir, os.pardir, "models"))
    return os.path.join(base, ROOT_NAME)


def _read(path):
    """优先用 ComfyUI 的加载器（兼容 safetensors 与 torch），否则退回原生。"""
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
            return load_file(path, device="cpu")
        except Exception:
            pass
    try:
        from comfy.utils import load_torch_file
        return load_torch_file(path, device=torch.device("cpu"))
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=True)


def _load_into(module, sd, name, strict=True):
    """把 state_dict 装进模块，并报告未命中 / 多余的键。"""
    missing, unexpected = module.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("rope_f")
               and not k.endswith("rope_h") and not k.endswith("rope_w")]
    if missing or unexpected:
        msg = (f"[fxai_vsr] {name} 键不匹配："
               f"缺 {len(missing)} 个 {missing[:6]}，"
               f"多 {len(unexpected)} 个 {unexpected[:6]}")
        if strict:
            raise RuntimeError(msg)
        print(msg)
    return module


def build_models(root=None, dtype=torch.bfloat16, device="cuda",
                 use_sparse=True, load_decoder=True):
    """构建 (dit, lq, decoder, context)。任何文件缺失都给出明确报错。"""
    root = root or model_root()
    need = [DIT_FILE, LQ_FILE, CTX_FILE] + ([TCD_FILE] if load_decoder else [])
    for f in need:
        p = os.path.join(root, f)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"缺少 {f}，请放到 {root}\n"
                f"（魔塔 AI-ModelScope/FlashVSR 可下全部四个权重文件）")

    cfg = DiTConfig(use_sparse=use_sparse)

    dit = StreamDiT(cfg)
    _load_into(dit, _read(os.path.join(root, DIT_FILE)), "DiT")
    dit = dit.to(device=device, dtype=dtype).eval()

    lq = LQProjector(in_dim=3, out_dim=cfg.dim, layer_num=1)
    _load_into(lq, _read(os.path.join(root, LQ_FILE)), "LQ 投影")
    lq = lq.to(device=device, dtype=dtype).eval()

    ctx = torch.load(os.path.join(root, CTX_FILE), map_location="cpu",
                     weights_only=True)
    ctx = ctx.to(device=device, dtype=dtype)

    dec = None
    if load_decoder:
        dec = TCDecoder(latent_channels=16 + 768)
        _load_into(dec, _read(os.path.join(root, TCD_FILE)), "TCDecoder")
        dec = dec.to(device=device, dtype=dtype).eval()

    return dit, lq, dec, ctx
