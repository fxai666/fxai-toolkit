# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权

"""
场景续接帧：段间引导帧的落盘读写工具。

被两个节点内部调用，不作为独立节点暴露：
  - FxAiMiniMaxVideoSaveV2：存本段视频时，把末尾 N 帧 + 对应音轨另存一份
  - FxAiMiniMaxImageToVideoV2：开「场景续接」后，按段序号读上一段的续帧

命名约定（读写两端同号，不做加减）：
  目录固定 fxai/video/scene_relay/，全局共用。
  第 N 段存 → scene_relay/<N+1>.pt  （给下一段用）
  第 N 段读 → scene_relay/<N>.pt    （上一段存的）
  第 0 段读 000.pt 不存在 → 自动跳过。

用 .pt 而不是 mp4：内存张量直接落盘，零损失、无 YUV 往返，音轨同存。
N 帧对齐到 H3 合法网格 17k+5（5, 22, 39, 56...）。
"""

import os

import torch

from fxai_minimax_video_save_v2 import get_video_dir

RELAY_SUBDIR = "scene_relay"
RELAY_FRAMES = 22


def _relay_path(段序号):
    # get_video_dir 带子目录参数时会 makedirs
    target = get_video_dir(RELAY_SUBDIR)
    return os.path.join(target, "%03d.pt" % int(段序号))


def _align_h3(n):
    """向下对齐到 H3 合法网格 17k+5：5, 22, 39, 56..."""
    n = max(5, int(n))
    while n % 17 != 5:
        n -= 1
    return n


def save_relay(frames, audio, 段序号, fps=24, count=RELAY_FRAMES):
    """把本段末尾 count 帧连同音轨存成 <段序号+1>.pt。返回路径，失败返回空串。"""
    if frames is None or frames.shape[0] == 0:
        return ""
    count = _align_h3(count)
    count = min(count, frames.shape[0])
    if count < 5:
        print("[凤希AI] 场景续接帧：本段不足 5 帧，跳过续帧")
        return ""
    tail = frames[-count:].detach().cpu()

    tail_audio = None
    if isinstance(audio, dict) and "waveform" in audio:
        sr = int(audio.get("sample_rate") or 32000)
        samples = int(round(count / float(fps) * sr))
        waveform = audio["waveform"]
        wav = waveform[..., -samples:].detach().cpu() if waveform.shape[-1] > samples else waveform.detach().cpu()
        tail_audio = {"waveform": wav, "sample_rate": sr}

    # +1：这份是给下一段（序号 N+1）用的，读端直接用当前段序号
    path = _relay_path(int(段序号) + 1)
    torch.save({"frames": tail, "audio": tail_audio, "fps": int(fps)}, path)
    print("[凤希AI] 场景续接帧：第 %d 段存 %d 帧（%s音轨）→ %s"
          % (int(段序号) + 1, tail.shape[0], "带" if tail_audio else "无", path))
    return path


def load_relay(段序号, fps=24):
    """读当前段序号对应的续帧（上一段存的）。返回 (frames, audio)，没命中返回 (None, None)。"""
    if 段序号 < 0:
        return None, None

    path = _relay_path(段序号)
    if not os.path.isfile(path):
        print("[凤希AI] 场景续接帧：续帧不存在，跳过引导：" + path)
        return None, None

    payload = torch.load(path, map_location="cpu", weights_only=False)
    frames = payload["frames"]
    audio = payload.get("audio")
    if frames is None or frames.shape[0] == 0:
        print("[凤希AI] 场景续接帧：续帧文件无画面，跳过引导：" + path)
        return None, None

    print("[凤希AI] 场景续接帧：第 %d 段读到上一段 %d 帧（%s音轨）← %s"
          % (int(段序号) + 1, frames.shape[0], "带" if audio else "无", path))
    return frames, audio
