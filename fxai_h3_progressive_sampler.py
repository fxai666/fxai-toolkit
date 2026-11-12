# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权

"""
凤希AI - H3 渐进采样（低清前缀 + 抬升 + 高清收尾）

把现在「一采 → latent 放大 → 二采」的三段式双采压进一个节点，关键在于**续采不重复加噪**：

1. 前缀段：在 sigmas[:k+1] 上采样。KSAMPLER 收尾时会做 inverse_noise_scaling(sigmas[k])。
   H3 的视频采样基类是 ModelSamplingAV + CONST，其 inverse_noise_scaling 是 latent / (1 - sigma)；
2. 尾段：用**全零 noise** 从 sigmas[k:] 续跑。CONST 的 noise_scaling 是
   sigma * noise + (1 - sigma) * latent_image，零 noise 下等于 (1 - sigma) * latent，
   正好把上一步的 / (1 - sigma) 精确抵消，续跑起点就是真实的 sigma_k 噪声潜变量。

所以总步数仍是 N 步、总前向仍是 N 次，与「从头到尾一次跑完」等价（浮点往返误差除外）。
这也意味着：相同的 sigmas 与采样器下，本节点与单段采样的输出应当一致，可作为验收手段。

抬升复用 fxai_minimax_upscaler 的权重与网络定义（LBH-123-AI），不复制模型结构、不改老节点。

锚点修正（rho）暂未实现：需先按 SelfLift 论文 arXiv:2609.02036 确定 high-risk 位置判据，
在此之前不提供服务。面板开关打开会**直接报错**，不做静默降级。
"""

import gc

import torch
import torch.nn.functional as F

import comfy.sample
import comfy.samplers
import comfy.utils

try:
    import comfy.model_management as mm
    HAS_COMFY_MM = True
except ImportError:
    HAS_COMFY_MM = False

try:
    from fxai_minimax_upscaler import (
        scan_models, load_model, _make_norm_tensors, VAE_DOWNSAMPLE,
    )
except ImportError:
    from .fxai_minimax_upscaler import (
        scan_models, load_model, _make_norm_tensors, VAE_DOWNSAMPLE,
    )

try:
    import fxai_task_store
    HAS_TASK_STORE = True
except Exception:
    HAS_TASK_STORE = False

# 与放大器保持一致：像素边长按 32 对齐，潜空间边长按 32/16 = 2 对齐
_PIXEL_ALIGN = 32
_LATENT_UNIT = _PIXEL_ALIGN // VAE_DOWNSAMPLE

_LIFT_MODES = ["学习型放大", "最近邻", "双线性"]


def _broadcast(event, payload):
    if not HAS_TASK_STORE:
        return
    try:
        fxai_task_store.broadcast(event, payload)
    except Exception as e:
        print(f"[凤希AI] 渐进采样广播失败：{e}")


def _target_hw(h, w, 倍数):
    """目标潜空间尺寸，按 32 像素对齐（潜空间步长 2）。"""
    h2 = max(_LATENT_UNIT, int(round(h * 倍数 / _LATENT_UNIT)) * _LATENT_UNIT)
    w2 = max(_LATENT_UNIT, int(round(w * 倍数 / _LATENT_UNIT)) * _LATENT_UNIT)
    return h2, w2


def _lift_latent(samples, mode, 倍数, 模型名, 时间分块, 释放显存):
    """把潜变量抬到目标尺寸。学习型放大复用 fxai_minimax_upscaler 的加载与前向。"""
    was_4d = samples.dim() == 4
    s = samples.unsqueeze(2) if was_4d else samples
    b, c, t, h, w = s.shape
    h2, w2 = _target_hw(h, w, 倍数)

    if (h2, w2) == (h, w):
        return samples

    if mode == "最近邻":
        out = F.interpolate(s, size=(t, h2, w2), mode="nearest")
    elif mode == "双线性":
        out = F.interpolate(s, size=(t, h2, w2), mode="trilinear", align_corners=False)
    else:
        if not 模型名 or 模型名.startswith("("):
            raise ValueError(
                "「学习型放大」需要选择放大模型。请把 H3 潜空间放大权重放入 "
                "models/latent_upscale_models/，或把「抬升方式」改成「最近邻」/「双线性」。"
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        up = load_model(模型名, device)
        up_dtype = next(up.parameters()).dtype
        norm_mean, norm_std = _make_norm_tensors(device, up_dtype)

        src_dtype = s.dtype
        work = s.to(device=device, dtype=up_dtype)
        with torch.inference_mode():
            work = (work - norm_mean) / norm_std
            out = up(work, scale=倍数, target_size=(t, h2, w2), enable_chunking=时间分块)
            out = out * norm_std + norm_mean
        out = out.to(device=samples.device, dtype=src_dtype)
        del work

        if device == "cuda" and 释放显存:
            up.to("cpu", non_blocking=True)
            if HAS_COMFY_MM:
                mm.soft_empty_cache()
            else:
                torch.cuda.empty_cache()
            gc.collect()

    return out.squeeze(2) if was_4d else out


class FxAiH3ProgressiveSampler:
    """凤希AI - H3 渐进采样"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT", {"tooltip": "低清画布。例如想要 768→1536，这里给 768 的潜变量，放大倍数给 2"}),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS", {"tooltip": "整条调度。节点会在「低清前缀步数」处切一刀，前后共用这一组 sigmas"}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "低清前缀步数": ("INT", {"default": 6, "min": 1, "max": 10000,
                                   "tooltip": "前多少步在低清画布上跑，剩下的抬升后在目标尺寸跑"}),
                "抬升方式": (_LIFT_MODES, {"default": "学习型放大"}),
                "放大倍数": ([1, 2, 3, 4], {"default": 2, "tooltip": "1 = 不抬升，等于纯合并调度，用于验证数值一致性"}),
            },
            "optional": {
                "放大模型": (scan_models(), {"tooltip": "仅「学习型放大」使用。模型与参考代码由 https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler 提供"}),
                "时间分块": ("BOOLEAN", {"default": True}),
                "释放显存": ("BOOLEAN", {"default": True}),
                "启用锚点修正": ("BOOLEAN", {"default": False,
                                      "tooltip": "论文 rho 锚点修正，尚未实现。打开会直接报错"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "run"
    CATEGORY = "凤希AI/视频"
    DESCRIPTION = ("渐进分辨率采样：低清前缀 → 抬升 → 高清收尾，压在一条调度里，"
                   "总步数不变、不额外多一次前向。取代「一采 + 放大 + 二采」的三段式接线。")

    def run(self, model, positive, negative, latent, sampler, sigmas, cfg, seed,
            低清前缀步数, 抬升方式, 放大倍数,
            放大模型="", 时间分块=True, 释放显存=True, 启用锚点修正=False):

        if 启用锚点修正:
            raise RuntimeError(
                "锚点修正（rho）尚未实现。它需要先按 SelfLift 论文 arXiv:2609.02036 确定 "
                "high-risk 位置的判据，在判据落地前不提供该能力，避免出现「开关打开了但实际没生效」的静默失败。"
                "请关闭「启用锚点修正」再运行。"
            )

        if "noise_mask" in latent:
            raise ValueError(
                "渐进采样暂不支持 noise_mask：抬升后掩码语义会变，无法保证正确。"
                "如需保留音频或局部区域，请先用现有双采接线。"
            )

        if not torch.is_tensor(sigmas):
            raise ValueError("sigmas 输入必须是张量，请接 ManualSigmas / BasicScheduler 等输出")

        n_steps = int(sigmas.shape[0]) - 1
        if n_steps < 2:
            raise ValueError(f"调度至少需要 2 步才能切分，当前只有 {n_steps} 步")

        k = int(低清前缀步数)
        if k >= n_steps:
            print(f"[凤希AI] 低清前缀步数 {k} 不小于总步数 {n_steps}，已收敛为 {n_steps - 1}（至少留 1 步高清收尾）")
            k = n_steps - 1
        if k < 1:
            k = 1

        x_low = latent["samples"]
        t, h, w = x_low.shape[-3:]
        h2, w2 = _target_hw(h, w, 放大倍数)

        print(f"[凤希AI] 渐进采样 - 总 {n_steps} 步：低清 {k} 步 @ ({h},{w}) → "
              f"抬升 {放大倍数}x（{抬升方式}）→ 高清 {n_steps - k} 步 @ ({h2},{w2})")

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # ---- 1. 低清前缀 ----
        noise = comfy.sample.prepare_noise(x_low, seed, latent.get("batch_index", None))
        x_k = comfy.sample.sample_custom(
            model, noise, cfg, sampler, sigmas[:k + 1], positive, negative,
            latent_image=x_low, noise_mask=None, callback=None,
            disable_pbar=disable_pbar, seed=seed,
        )
        del noise

        # ---- 2. 抬升 ----
        if 放大倍数 == 1:
            lifted = x_k
        else:
            _broadcast("upscale_start", {"message": f"渐进采样抬升：{放大倍数}x"})
            lifted = _lift_latent(x_k, 抬升方式, 放大倍数, 放大模型, 时间分块, 释放显存)
            del x_k
            _broadcast("upscale_done", {"message": f"渐进采样抬升完成：{放大倍数}x"})

        # ---- 3. 高清收尾：零噪声续跑，noise_scaling 抵消 inverse_noise_scaling ----
        zero_noise = torch.zeros_like(lifted)
        samples = comfy.sample.sample_custom(
            model, zero_noise, cfg, sampler, sigmas[k:], positive, negative,
            latent_image=lifted, noise_mask=None, callback=None,
            disable_pbar=disable_pbar, seed=seed,
        )
        del zero_noise, lifted

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out.pop("noise_mask", None)
        out["samples"] = samples
        return (out,)
