# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# MiniMax H3 核心补丁：不修改 ComfyUI 系统文件，也不复制官方 MiniMaxH3Model 的前向实现，
# 只替换模型基类并叠加下面两处改动。官方的 _forward / _run_blocks / 块循环全部原样生效，
# 核心升级后本补丁无需跟随同步。
# 1) MiniMaxH3.extra_conds：把 keyframe 视觉 latent 对齐到布局目标空间网格，
#    并让 ref 的 latent_h/w/t 跟张量一致，避免 layout 行数与 patchify 分叉。
# 2) MiniMaxH3Patch.__init__：把 noise_scaling 改成 EPS 风格，保留写入潜空间的过渡帧
#    作为采样起点（FLOW 完整去噪会把 init 潜空间清成纯噪声）。
#
# 块级加速（FxAiMiniMaxBlockCache 的块缓存）走官方 ("double_block", i) 钩子实现，
# 见 fxai_minimax_block_cache.py。
#
# 容错约定：官方核心若改了类名或挪走 reshape_sigma，本补丁静默退化为官方原行为并打印一次提示，
# 不会阻断 fxai-toolkit 其余节点加载。noise_scaling 不加兜底——它每步都调用且只做三行张量运算，
# 出错说明输入本身有问题，吞掉只会把画质问题藏起来。

import math

import torch

import comfy.model_base
import comfy.model_sampling

_H3_BASE = getattr(comfy.model_base, "MiniMaxH3", None)
# 官方 CONSTAT/FLOW 的 reshape_sigma；改名或移除时我们无法安全地重算 sigma，保持官方实现。
_reshape_sigma = getattr(comfy.model_sampling, "reshape_sigma", None)


def _align_cond_shapes(kwargs):
    """把 keyframe / ref 的 latent 对齐到布局目标空间网格，并同步 ref 的 latent_h/w/t。"""
    latent_shapes = kwargs.get("latent_shapes", None)
    if latent_shapes is None or len(latent_shapes) <= 1:
        return
    vs = latent_shapes[0]
    th = (vs[3] + 1) // 2 * 2
    tw = (vs[4] + 1) // 2 * 2
    for kf in kwargs.get("minimax_keyframes") or ():
        z = kf.get("latent")
        if z is None:
            continue
        if z.shape[-2] != th or z.shape[-1] != tw:
            kf["latent"] = torch.nn.functional.interpolate(
                z.to(torch.float32), size=(z.shape[2], th, tw),
                mode="trilinear", align_corners=False).to(z.dtype)
    for r in kwargs.get("minimax_refs") or ():
        z = r.get("latent")
        if z is None:
            continue
        r["latent_h"] = z.shape[-2]
        r["latent_w"] = z.shape[-1]
        if r.get("kind") in ("video", "video_audio"):
            r["latent_t"] = z.shape[2]


if _H3_BASE is None:
    print("【凤希AI】跳过 MiniMax H3 核心补丁：comfy.model_base 中已无 MiniMaxH3，"
          "核心可能重命名了该类，本次按官方原行为运行。")
else:
    class MiniMaxH3Patch(_H3_BASE):
        def __init__(self, model_config, model_type=None, device=None):
            # 不指定 model_type，跟随核心默认值：新版核心用 FLOW_AV（ModelSamplingAV，
            # 含 audio_scale），旧版核心用 FLOW，避免采样时缺 audio_scale 崩溃。
            if model_type is None:
                super().__init__(model_config, device=device)
            else:
                super().__init__(model_config, model_type, device=device)
            if _reshape_sigma is None:
                return
            # 保留写入潜空间的过渡帧作为采样起点：FLOW 完整去噪时 (1-sigma)*latent 权重
            # 在 sigma=1 处为 0，会把已写入的 init 潜空间清成纯噪声。改成 EPS 风格
            # `sigma*noise + latent`：过渡帧区域（非零）从写入内容起步演化，其余全零
            # 区域起点不变，行为等同原逻辑。
            ms = self.model_sampling
            owner = self

            def noise_scaling(sigma, noise, latent_image, max_denoise=False):
                sigma = _reshape_sigma(sigma, noise.ndim)
                scale = getattr(ms, "noise_scale", 1.0)
                added = sigma * (scale * noise)
                # MiniMax H3 把音频流按 audio_scale 缩放后与视频流拼在同一张量的最后一维，
                # 所以这里必须分段处理，不能整条套用同一公式：
                #   视频段 → EPS 风格 `sigma*noise + latent`，保住写进去的过渡帧/K 帧；
                #   音频段 → 官方 `sigma*noise + (1-sigma)*latent`，flow 正常衰减。
                # 若对音频段也用 EPS 风格，audio_scale(shift/audio_shift=12/3=4) 放大的
                # 音频信号会以 1.0 而非 (1-sigma) 的权重进入采样起点，二采 sigma≈0.9 时
                # 相当于放大 1/(1-sigma)≈10 倍，3 步采样来不及拉回，成片音频会过载发闷/嗡嗡。
                shapes = getattr(owner, "latent_shapes", None)
                video_elems = math.prod(shapes[0][1:]) if shapes else 0
                if (not torch.is_tensor(latent_image) or len(shapes or ()) < 2
                        or latent_image.shape[-1] <= video_elems):
                    # 只有视频流（或分段信息与张量对不上，例如 context window 改写过
                    # latent_shapes）时整条按 EPS 风格处理：与旧行为一致。
                    return added + latent_image
                out = added + (1.0 - sigma) * latent_image
                out[..., :video_elems] = (added + latent_image)[..., :video_elems]
                return out

            ms.noise_scaling = noise_scaling

        def extra_conds(self, **kwargs):
            # 尺寸兜底：布局按目标 even-rounded 网格计行，_cond_video_rows 按 keyframe
            # 自身 H/W patchify，分叉会 shape mismatch。放大后优先用
            # FxAiMiniMaxResampleCond 按实际宽高 VAE 重编码；此处仅在仍不一致时
            # 三线性对齐，避免漏接二采条件直接崩采样。
            # 对齐本身出错时退回官方原始条件（可能再崩在采样，但至少不是补丁引入的崩溃）。
            try:
                _align_cond_shapes(kwargs)
            except Exception as exc:
                print(f"【凤希AI】MiniMax H3 条件尺寸对齐已跳过，按官方原始条件继续（原因：{exc!r}）")
            return super().extra_conds(**kwargs)

    def apply_patch():
        comfy.model_base.MiniMaxH3 = MiniMaxH3Patch

    apply_patch()
