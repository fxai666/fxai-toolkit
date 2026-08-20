# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# MiniMax H3 核心补丁：不修改 ComfyUI 系统文件，通过替换模块引用注入修复。
# 1) MiniMaxH3.extra_conds：keyframes（首尾帧）与 refs（参考图/音频）的视觉条件共存，
#    官方逻辑里 refs 会覆盖 keyframes 的 cond_video_latents，导致首尾帧锚定失效。
# 2) MiniMaxH3Model._forward / _run_blocks：支持 ("block_loop", 0) 钩子，
#    供 FxAiMiniMaxBlockCache 块缓存加速节点使用。
#
# 音频参考/音频输出缩放一律走官方实现：PackedLayout 推进参考音频时间轴，
# 音频速度场缩放由官方 forward 的 audio_scale 机制处理，这里不重写。

import torch

import comfy.model_base
import comfy.model_management
import comfy.model_prefetch
import comfy.model_sampling
import comfy.ldm.common_dit
import comfy.ldm.minimax.model as h3

# 兼容新旧官方内核：官方有 time_shift_sigma 就用官方的，没有才用本地兜底。
if hasattr(h3, "time_shift_sigma"):
    time_shift_sigma = h3.time_shift_sigma
else:
    def time_shift_sigma(sigma, from_shift, to_shift):
        base = sigma / (from_shift + sigma * (1.0 - from_shift))
        return to_shift * base / (1.0 + (to_shift - 1.0) * base)


class MiniMaxH3Patch(comfy.model_base.MiniMaxH3):
    def __init__(self, model_config, model_type=None, device=None):
        # 不指定 model_type，跟随核心默认值：新版核心用 FLOW_AV（ModelSamplingAV，
        # 含 audio_scale），旧版核心用 FLOW，避免采样时缺 audio_scale 崩溃。
        if model_type is None:
            super().__init__(model_config, device=device)
        else:
            super().__init__(model_config, model_type, device=device)
        # 保留写入潜空间的过渡帧作为采样起点：FLOW 完整去噪时 (1-sigma)*latent 权重
        # 在 sigma=1 处为 0，会把已写入的 init 潜空间清成纯噪声。改成 EPS 风格
        # `sigma*noise + latent`：过渡帧区域（非零）从写入内容起步演化，其余全零
        # 区域起点不变，行为等同原逻辑。
        ms = self.model_sampling

        def noise_scaling(sigma, noise, latent_image, max_denoise=False):
            sigma = comfy.model_sampling.reshape_sigma(sigma, noise.ndim)
            scale = getattr(ms, "noise_scale", 1.0)
            return sigma * (scale * noise) + latent_image

        ms.noise_scaling = noise_scaling

    def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
        # 掩码锁死区（过渡帧写入的 t 前 k 步）每步按 FLOW 语义注入：
        # sigma 大时接近噪声、sigma→0 收敛到干净过渡帧 latent，避免
        # latent 权重恒为 1 导致锁死区色彩/内容过度保留而失衡。
        sigma = comfy.model_sampling.reshape_sigma(sigma, noise.ndim)
        scale = getattr(self.model_sampling, "noise_scale", 1.0)
        return sigma * (scale * noise) + (1.0 - sigma) * latent_image

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        keyframes = kwargs.get("minimax_keyframes", None)
        refs = kwargs.get("minimax_refs", None)
        if keyframes is not None or refs is not None:
            payload = out["minimax_payload"].cond
            # keyframes 与 refs 的视觉条件共存：官方 refs 会覆盖 keyframes 的
            # cond_video_latents，这里拼接而不是覆盖。视频按 keyframe 在前、
            # 参考块在列表序；音频独立列表，同样拼接不覆盖。
            payload["cond_video_latents"] = (
                [kf["latent"] for kf in (keyframes or []) if kf.get("latent") is not None]
                + [r["latent"] for r in (refs or []) if "latent" in r])
            payload["cond_audio_latents"] = (
                [kf["audio_latent"] for kf in (keyframes or []) if kf.get("audio_latent") is not None]
                + [r["audio_latent"] for r in (refs or []) if r.get("audio_latent") is not None])
        return out


def _run_blocks(self, h, t_emb, mod_segments, rope_freqs, transformer_options, start=0, end=None):
    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    end = len(self.blocks) if end is None else end
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks[start:end]), h.device, transformer_options)
    for i in range(start, end):
        block = self.blocks[i]
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, h.device, block)
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                     transformer_options=args["transformer_options"])}
            h = blocks_replace[("double_block", i)](
                {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                 "transformer_options": transformer_options},
                {"original_block": block_wrap})["img"]
        else:
            h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
    if prefetch_queue is not None:
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, h.device, None)
    return h


def _patched_forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
    video_x, audio_x = x[0], x[1]
    orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)
    if video_x.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    payload = minimax_payload or {}
    device = video_x.device
    dtype = context.dtype  # compute dtype

    latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    layout = payload.get("layout")
    if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
        layout = h3.PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                                 keyframes=payload.get("keyframes"),
                                 refs=payload.get("refs"),
                                 frame_count=payload.get("frame_count"))

    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio))
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))

    vis_aug = float(payload.get("visual_cond_noise_aug", h3.VISUAL_COND_TIMESTEP))
    aud_aug = float(payload.get("audio_cond_noise_aug", h3.AUDIO_COND_TIMESTEP))
    has_vis_cond = any(k in ("cond", "ref_img") for _, _, k in layout.segments)
    has_aud_cond = any(k == "ref_audio" for _, _, k in layout.segments)
    seg_t = {"text": t_v, "video": t_v, "audio": t_a,
             "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
             "ref_audio": max(t_a, aud_aug)}
    unique_t = sorted({t_v, t_a} | ({seg_t["cond"]} if has_vis_cond else set())
                      | ({seg_t["ref_audio"]} if has_aud_cond else set()))
    t_row = {t: i for i, t in enumerate(unique_t)}
    seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "ref_audio": 2}

    text_tags = payload.get("text_token_tags")
    mod_segments = []
    for a, b, kind in layout.segments:
        row_base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            tags = text_tags.view(-1).tolist()
            run_start = 0
            for i in range(1, b - a + 1):
                if i == b - a or tags[i] != tags[run_start]:
                    mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                    run_start = i
        else:
            mod_segments.append((a, b, row_base + seg_tag[kind]))

    img_update = layout.img_update.to(device)
    audio_update = layout.audio_update.to(device)
    video_rows = h3.patchify_video(video_x.to(torch.float32), self.patch_size)
    audio_rows = h3.pack_audio(audio_x.to(torch.float32))
    cond_video_rows = self._cond_video_rows(payload, device)
    cond_audio_rows = self._cond_audio_rows(payload, device)

    all_video_rows = video_rows
    if cond_video_rows is not None:
        all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
        all_video_rows[~img_update] = cond_video_rows
        all_video_rows[img_update] = video_rows
    all_audio_rows = audio_rows
    if cond_audio_rows is not None:
        all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
        all_audio_rows[~audio_update] = cond_audio_rows
        all_audio_rows[audio_update] = audio_rows

    video_embed = self.video_patch_proj(all_video_rows).to(dtype)
    audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
    text_states = context[0]
    if text_states.shape[-1] != self.hidden_size:
        text_states = self.token_refiner(self.condition_proj(text_states),
                                         transformer_options=transformer_options)

    h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
    voff = aoff = 0
    for a, b, kind in layout.segments:
        n = b - a
        if kind == "text":
            h[a:b] = text_states
        elif kind in ("cond", "ref_img", "video"):
            h[a:b] = video_embed[voff:voff + n]
            voff += n
        else:  # ref_audio / audio
            h[a:b] = audio_embed[aoff:aoff + n]
            aoff += n

    t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if self.use_adaln_curves:
        table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
    else:
        t_emb = self.time_embedder(t_vals).to(dtype)

    rope_freqs = h3.rope_rotation_table(self.rope_freqs(layout.position_ids, device), dtype)

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    cache_ranges = [(a, b) for a, b, kind in layout.segments if kind in ("audio", "video")]
    if ("block_loop", 0) in blocks_replace:
        def block_loop_wrap(args):
            return {"img": self._run_blocks(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                            args["transformer_options"], args.get("start", 0), args.get("end"))}
        h = blocks_replace[("block_loop", 0)](
            {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
             "transformer_options": transformer_options, "cache_ranges": cache_ranges, "block_count": len(self.blocks)},
            {"original_block": block_loop_wrap})["img"]
    else:
        h = self._run_blocks(h, t_emb, mod_segments, rope_freqs, transformer_options)

    video_seg = next((a, b, t_row[seg_t["video"]]) for a, b, k in layout.segments if k == "video")
    audio_seg = next((a, b, t_row[seg_t["audio"]]) for a, b, k in layout.segments if k == "audio")
    v, a = self.final_layer(h, t_emb, video_seg, audio_seg)

    video_out = h3.unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
    video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
    audio_out = h3.unpack_audio(a)

    return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]


def apply_patch():
    comfy.model_base.MiniMaxH3 = MiniMaxH3Patch
    h3.MiniMaxH3Model._run_blocks = _run_blocks
    h3.MiniMaxH3Model._forward = _patched_forward

apply_patch()
