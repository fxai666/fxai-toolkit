# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# 视频分段加载：对应「MiniMax音频分段加载」的视频版。
# 分段器只给路径和每段时长，这里按「当前索引」把前面各段时长累加成帧数，
# 只解码这一段——整条视频不进内存。
#
# 音频是同一段画面的音轨，用同一套帧号换算切，配音不会错位。

import datetime

from fxai_task_store import broadcast

from fxai_video_utils import (
    decode_audio_range,
    decode_frame_range,
    probe_video,
    resolve_video_path,
    round_half_up,
)

# 单段超过这个秒数基本可以断定上游传的是帧数，不是时长
MAX_SEGMENT_SECONDS = 600.0


class FxAiVideoSegmentLoad:
    CATEGORY = "凤希AI/视频"
    FUNCTION = "load_segment"

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT")
    RETURN_NAMES = ("视频帧", "音频", "本段帧数")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "视频路径": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "接「视频分段器V1」输出的视频路径（input 下的相对路径），\n"
                               "也可以手填完整路径或相对路径。"}),
                "分段时长列表": ("LIST", {"forceInput": True}),
                "当前索引": ("INT", {"default": 0, "min": 0,
                    "tooltip": "第几段，从 0 开始。接循环控制器逐次递增。"}),
                "起始帧": ("INT", {"default": 0, "min": 0,
                    "tooltip": "分段器掐头后的起点帧，接「视频分段器V1」的「起始帧」。\n"
                               "没掐头就是 0。不接这个，跳过初始段之后画面会从第 0 帧切起。"}),
            }
        }

    def load_segment(self, 视频路径, 分段时长列表, 当前索引=0, 起始帧=0):
        path = resolve_video_path(视频路径)
        durations = [float(value) for value in 分段时长列表]
        total = len(durations)
        if total == 0:
            raise ValueError("分段时长列表是空的，检查「视频分段器V1」有没有算出分段")
        if max(durations) > MAX_SEGMENT_SECONDS:
            raise ValueError(
                f"分段时长列表疑似传入了帧数而不是秒数（最大一段 {max(durations):.0f} > "
                f"{MAX_SEGMENT_SECONDS:.0f} 秒）。请接「视频分段器V1」的「分段时长列表」。")

        index = int(当前索引)
        if index < 0 or index >= total:
            raise ValueError(f"当前索引 {index} 超出范围：共 {total} 段，有效 0 ~ {total - 1}")

        info = probe_video(path)
        fps = info["fps"]
        # 和分段器同一套取整（四舍五入到帧），秒 → 帧来回一趟不丢帧
        counts = [max(1, round_half_up(value * fps)) for value in durations]
        # 起点 = 掐头偏移 + 前面各段长度；不加上游的偏移，跳过初始段后会从第 0 帧切起
        start_frame = max(0, int(起始帧)) + sum(counts[:index])
        start_frame = min(start_frame, max(0, info["frames"] - 1))

        # 末段可能被四舍五入顶出去，不超过视频实际帧数
        take = max(1, min(counts[index], info["frames"] - start_frame))

        print(f"✅ [凤希AI] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"视频分段加载：第 {index + 1}/{total} 段 · 帧 {start_frame}~{start_frame + take - 1}"
              f"（{take} 帧 / {durations[index]:.3f} 秒）")

        frames = decode_frame_range(path, start_frame, take, fps, info["width"], info["height"])
        audio = decode_audio_range(path, start_frame, take, fps)
        actual = int(frames.shape[0])
        if actual != take:
            print(f"[凤希AI] 第 {index + 1} 段计划 {take} 帧，实际解出 {actual} 帧"
                  f"（视频尾段不足，或源帧率与容器标称不一致）")

        broadcast("scene_executing", {
            "current": index + 1,
            "total": total,
            "duration": durations[index],
            "frames": actual,
            "message": f"开始渲染第 {index + 1}/{total} 个场景",
        })

        return (frames, audio, actual)
