# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# MiniMax H3 核心补丁：不修改 ComfyUI 系统文件，通过替换模块引用注入两处修复。
# 1) MiniMaxH3.extra_conds：keyframes（首尾帧）与 refs（参考图/音频）的视觉条件共存，
#    官方逻辑里 refs 会覆盖 keyframes 的 cond_video_latents，导致首尾帧锚定失效。
# 2) PackedLayout：standalone 参考音频不推进时间轴，保证外置音频口型时序正确。
#
# 替换引用后，系统加载 H3 checkpoint 时（supported_models -> model_base.MiniMaxH3）
# 会实例化这里的子类，模型 forward 里硬编码的 PackedLayout 也指向这里的子类。

import torch

import comfy.model_base
import comfy.ldm.minimax.model as h3


class H3MiniMaxH3(comfy.model_base.MiniMaxH3):
    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        keyframes = kwargs.get("minimax_keyframes", None)
        if keyframes is not None:
            payload = out["minimax_payload"].cond
            refs = payload.get("refs", None) or []
            payload["cond_video_latents"] = [kf["latent"] for kf in keyframes] + [
                r["latent"] for r in refs if "latent" in r]
        return out


class H3PackedLayout(h3.PackedLayout):
    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None, frame_count=None):
        frame, w_grid = h3._frame_grid(latent_h, latent_w)
        frame_rows = frame.shape[0]

        segments = [("text", text_len)]  # (kind, n_rows)
        g = torch.zeros(text_len, 3, dtype=torch.float64)
        g[:, 0] = torch.arange(text_len, dtype=torch.float64)
        pos = [g]  # per segment: [n, 3] float64 (t, h, w)

        img_pos, img_update = [], []
        audio_pos, audio_update = [], []
        cursor = text_len
        row = text_len

        if keyframes:
            # fl2va: keyframe cond rows right after text, sharing the target spatial grid
            for kf in keyframes:
                pixel_index = kf["resolved_frame_index"]
                if pixel_index == 0:
                    cond_t = float(text_len)
                elif frame_count is not None and pixel_index == frame_count - 1:
                    cond_t = float(text_len) + sum(h3._video_t_spans(latent_t)) - h3.FRAME_RESCALE
                else:
                    raise ValueError("only first/last keyframe anchors are supported")
                g = torch.empty(frame_rows, 3, dtype=torch.float64)
                g[:, 0] = cond_t
                g[:, 1:] = frame
                segments.append(("cond", frame_rows))
                pos.append(g)
                img_pos.append(torch.arange(row, row + frame_rows))
                img_update.append(torch.zeros(frame_rows, dtype=torch.bool))
                row += frame_rows

        target_audio_w = (float(w_grid[0]), float(w_grid[-1]))
        if refs:
            cursor = float(text_len)
            for blk in refs:
                kind = blk["kind"]
                if kind == "image":
                    r_frame, _ = h3._frame_grid(blk["latent_h"], blk["latent_w"])
                    n = r_frame.shape[0]
                    g = torch.empty(n, 3, dtype=torch.float64)
                    g[:, 0] = cursor
                    g[:, 1:] = r_frame
                    segments.append(("ref_img", n))
                    pos.append(g)
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    cursor += 1.0
                elif kind == "audio":
                    # standalone reference audio does not advance the timeline
                    rt = blk["ref_audio_t"]
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(h3._audio_grid(cursor, rt, *target_audio_w))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                elif kind in ("video", "video_audio"):
                    # the block's audio rows pack immediately before its video
                    # rows, both sharing the cursor origin
                    rt = blk["ref_audio_t"]
                    vt = blk["latent_t"]
                    r_frame, r_w_grid = h3._frame_grid(blk["latent_h"], blk["latent_w"])
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(h3._audio_grid(cursor, rt, float(r_w_grid[0]), float(r_w_grid[-1])))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                    n = vt * r_frame.shape[0]
                    segments.append(("ref_img", n))
                    pos.append(h3._video_grid(vt, r_frame, cursor))
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    cursor += max(float(rt), sum(h3._video_t_spans(vt)))

        # target audio then target video, always the last two segments
        segments.append(("audio", audio_t * 2))
        pos.append(h3._audio_grid(cursor, audio_t, *target_audio_w))
        audio_pos.append(torch.arange(row, row + audio_t * 2))
        audio_update.append(torch.ones(audio_t * 2, dtype=torch.bool))
        row += audio_t * 2

        n_video = latent_t * frame_rows
        segments.append(("video", n_video))
        pos.append(h3._video_grid(latent_t, frame, cursor))
        img_pos.append(torch.arange(row, row + n_video))
        img_update.append(torch.ones(n_video, dtype=torch.bool))
        row += n_video

        self.seq_len = row
        self.position_ids = torch.cat(pos)  # [S, 3] float64
        self.img_pos = torch.cat(img_pos)
        self.img_update = torch.cat(img_update)
        self.audio_pos = torch.cat(audio_pos)
        self.audio_update = torch.cat(audio_update)
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        # contiguous segment table (start, stop, kind)
        # kinds: text / cond / ref_img / ref_audio / audio / video
        # the packed sequence is uniform per segment in (modality tag, timestep class),
        # except the text span (tag runs resolved at forward time from the presentation tags)
        seg_abs = []
        off = 0
        for kind, n in segments:
            seg_abs.append((off, off + n, kind))
            off += n
        self.segments = seg_abs


def apply_patch():
    comfy.model_base.MiniMaxH3 = H3MiniMaxH3
    h3.PackedLayout = H3PackedLayout

apply_patch()
