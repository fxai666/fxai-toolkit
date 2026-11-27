# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权

"""
场景续接帧：段间引导帧的落盘读写工具。

被两个节点内部调用，不作为独立节点暴露：
  - FxAiMiniMaxVideoSaveV2：存本段视频时，把末尾 N 帧 + 对应音轨另存一份
  - FxAiMiniMaxImageToVideoV2：按段序号读上一段的续帧（文件在就接，不在就硬切）

命名约定（写端 +1、读端不加，两边按段序号严格配对）：
  目录固定 fxai/video/scene_relay/，全局共用。
  第 N 段写 → scene_relay/<N+1>.pt  （给下一段用）
  第 N 段读 → scene_relay/<N>.pt    （上一段写的）
  第 0 段读 000.pt 不存在 → 静默跳过。

两个关键约定（别改，改了就会错乱）：
  1. **判断只在写端**：表格里「场景续接」勾在第 N 行 = 第 N 段与第 N+1 段之间要续。
     勾 → 写 <N+1>.pt；不勾 → 删 <N+1>.pt。读端只看文件在不在，不重复判断，
     两端口径只有一个，不会出现「勾了不接 / 没勾却接上」。
  2. 文件里带 seg（本文件供第几段读），读端校验不一致就跳过，
     防止手工改名或跨批次残留的错位文件被当成上一段。

用 .pt 而不是 mp4：内存张量直接落盘，零损失、无 YUV 往返，音轨同存。
N 帧对齐到 H3 合法网格 17k+5（5, 22, 39, 56...）。
"""

import os
import time

import torch

from fxai_minimax_video_save_v2 import get_video_dir

RELAY_SUBDIR = "scene_relay"
RELAY_FRAMES = 1   # V2 只用最后 1 帧锚第 0 帧；多存只增体积，不增效果


def _relay_path(段序号):
    # get_video_dir 带子目录参数时会 makedirs
    target = get_video_dir(RELAY_SUBDIR)
    return os.path.join(target, "%03d.pt" % int(段序号))


def clear_relay(段序号):
    """删掉本段本该给下一段留的续帧文件（<段序号+1>.pt）。

    「场景续接」不勾时调用：接缝标在本段，本段之后不要续，
    就把本段该留的接点删掉，下一段读不到自然硬切。与 save_relay 同号，各段各清各的。
    """
    path = _relay_path(int(段序号) + 1)
    if os.path.isfile(path):
        os.remove(path)
        print("[凤希AI] 场景续接关闭：清理残留续帧 %s" % path)


def _align_h3(n):
    """向下对齐到 H3 合法网格 17k+5：5, 22, 39, 56..."""
    n = max(5, int(n))
    while n % 17 != 5:
        n -= 1
    return n


def _flatten_luma(frames):
    """把续帧各帧的亮度拉到同一灰阶（取帧均值作基准），只消帧间差、不动整体色阶。

    续帧是上一段解码输出的末尾 N 帧，帧与帧之间仍有 ±1~2 灰阶的走势。
    原样存进 .pt 后，下一段 V2 读出来当 K 帧，这段走势就被带进新段开头，
    表现为接缝后几帧的亮度跳动。存盘前把每帧各自平移到共同均值即可消掉。
    """
    if frames is None or frames.shape[0] < 2:
        return frames
    g = frames.to(torch.float32)
    luma = 0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]
    base = float(luma.mean())                       # 全部帧的共同均值
    delta = base - luma.mean(dim=(1, 2))            # 每帧各自要补的灰阶 [T]
    out = g + delta.view(-1, 1, 1, 1)
    return out.clamp_(0.0, 1.0).to(frames.dtype)


def save_relay(frames, audio, 段序号, fps=24, count=RELAY_FRAMES, source=""):
    """把本段末尾 count 帧连同音轨存成 <段序号+1>.pt。返回路径，失败返回空串。

    source：本段成片路径，只取文件名记进 pt，读端打印出来供人工核对配对
            （“第 5 段读到的是第 4 段的 xxx.mp4” 一眼可见，错位马上暴露）。
    """
    if frames is None or frames.shape[0] == 0:
        return ""
    # 1 帧是合法档位（V2 只用最后 1 帧做锚点，1 帧 ≈ 12MB；5 帧 ≈ 2GB 级，白占盘）
    count = max(1, int(count))
    if count > 1:
        count = _align_h3(count)
    count = min(count, frames.shape[0])
    if count < 1:
        print("[凤希AI] 场景续接帧：本段没有可用帧，跳过续帧")
        return ""
    # 🔴 .clone() 一个都不能省：frames[-count:] 是切片**视图**，数据显示只有 count 帧，
    # 但底层 storage 仍指着整段视频（192 帧 ≈ 1.93GB）。.cpu() 在数据本来就在 CPU 时
    # 直接返回原对象、不复制，torch.save 又是照 storage 而非照视图写盘 ——
    # 结果是「只存 1 帧」却写出 1.94GB（实测 2026-11-28）。clone 强制切成独立小块（1 帧约 10MB）。
    tail = frames[-count:].detach().cpu().clone()
    # 亮度统一移到读入端做（测试阶段只在读入端调整），这里原样落盘 PT。
    # _flatten_luma 保留在此模块，供 V2 读入端 import 调用。

    tail_audio = None
    if isinstance(audio, dict) and "waveform" in audio:
        sr = int(audio.get("sample_rate") or 32000)
        samples = int(round(count / float(fps) * sr))
        waveform = audio["waveform"]
        # 同理必须 clone：切片音轨背后是整条 waveform，不切断会把整段音轨打包进 pt
        _wav_src = waveform[..., -samples:] if waveform.shape[-1] > samples else waveform
        wav = _wav_src.detach().cpu().clone()
        tail_audio = {"waveform": wav, "sample_rate": sr}

    # +1：这份是给下一段（序号 N+1）用的，读端直接用当前段序号
    path = _relay_path(int(段序号) + 1)
    torch.save({
        "frames": tail,
        "audio": tail_audio,
        "fps": int(fps),
        # 配对信息：seg=这份文件供第几段读，from_seg=哪一段写出来的，at=写入时刻
        "seg": int(段序号) + 1,
        "from_seg": int(段序号),
        "source": os.path.basename(str(source)) if source else "",
        "at": time.time(),
    }, path)
    print("[凤希AI] 场景续接帧：第 %d 段存 %d 帧（%s音轨）→ 给第 %d 段做接点 %s"
          % (int(段序号) + 1, tail.shape[0], "带" if tail_audio else "无",
             int(段序号) + 2, path))
    return path


def load_relay(段序号, fps=24):
    """读当前段序号对应的续帧（上一段写的）。返回 (frames, audio)，没命中返回 (None, None)。"""
    if 段序号 < 0:
        return None, None

    path = _relay_path(段序号)
    if not os.path.isfile(path):
        print("[凤希AI] 场景续接帧：续帧不存在，本段独立起头：" + path)
        return None, None

    payload = torch.load(path, map_location="cpu", weights_only=False)
    # 配对校验：文件自己写着「我是给第几段用的」，对不上就是错位文件（改名/跨批次），宁可不接
    seg = payload.get("seg")
    if seg is not None and int(seg) != int(段序号):
        print("[凤希AI] 场景续接帧：段号对不上（文件是第 %s 段的，本次要第 %d 段），跳过：%s"
              % (seg, int(段序号), path))
        return None, None

    frames = payload["frames"]
    audio = payload.get("audio")
    if frames is None or frames.shape[0] == 0:
        print("[凤希AI] 场景续接帧：续帧文件无画面，跳过引导：" + path)
        return None, None

    at = payload.get("at")
    when = ""
    if isinstance(at, (int, float)) and at > 0:
        when = "（%.0f 分钟前写入）" % max(0.0, (time.time() - at) / 60.0)
    from_seg = payload.get("from_seg")
    src = ("第 %d 段留下的" % (int(from_seg) + 1)) if isinstance(from_seg, (int, float)) else "上一段留下的"
    source = payload.get("source") or ""
    print("[凤希AI] 场景续接帧：第 %d 段读到%s %d 帧%s（%s音轨）%s← %s"
          % (int(段序号) + 1, src, frames.shape[0], when, "带" if audio else "无",
             ("[" + source + "] ") if source else "", path))
    return frames, audio
