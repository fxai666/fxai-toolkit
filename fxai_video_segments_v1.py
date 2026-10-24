# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# 视频分段器V1：和「音频分段器V2」同一套面板逻辑，区别是音频按采样点切、
# 视频按帧切。这里只算分段，不解码画面——输出视频路径 + 每段时长列表，
# 交给「视频分段加载」按索引逐段读，避免把整条视频塞进内存。
#
# 输出时长（秒）而不是帧数：下游「生成控制器」和 MiniMax 生成节点都按时长驱动，
# 帧数列表接上去会被当成秒数用，整条链路对不上。
#
# 视频文件用 input 目录下的相对路径（子目录/文件名），和音频分段器一致，
# 这样工作流换机器、走 API 调用都不用关心服务器上的物理路径。

import datetime

from fxai_video_utils import (
    H3_FPS,
    align_down_h3,
    align_down_ltx,
    build_frame_segments,
    decode_audio_range,
    format_fps,
    format_time,
    list_input_video_files,
    parse_keyframe_list,
    probe_video,
    rel_video_path,
    resolve_video_path,
    round_half_up,
    segment_frames,
)

INFO_MAX_LINES = 60
DURATION_DECIMALS = 5


def _step_detail(avg_dur, fps, model, step):
    """把「填的秒数」到「实际每段帧数」的换算过程写清楚。

    三种模型的对齐规则不一样，光看结果（比如填 10 秒却切成 226 帧）
    会以为算错了，所以中间步骤一并输出。
    """
    name = str(model or "").strip().lower()
    if name in ("minimax", "minimax h3", "h3"):
        raw_h3 = round_half_up(avg_dur * H3_FPS)
        h3 = align_down_h3(raw_h3)
        return (f"  换算：{avg_dur:.2f} 秒 × H3 {H3_FPS}fps = {raw_h3} 帧"
                f" → 17k+5 对齐 {h3} 帧 → 回源 {format_fps(fps)}fps 得 {step} 帧/段"
                f"（{step / fps:.3f} 秒）")
    if name == "ltx":
        raw = round_half_up(avg_dur * fps)
        return (f"  换算：{avg_dur:.2f} 秒 × {format_fps(fps)}fps = {raw} 帧"
                f" → 8k+1 向下对齐 {align_down_ltx(raw)} 帧/段"
                f"（{step / fps:.3f} 秒）")
    return (f"  换算：{avg_dur:.2f} 秒 × {format_fps(fps)}fps = {step} 帧/段"
            f"（{step / fps:.3f} 秒，四舍五入到帧，不做对齐）")


def frames_to_durations(segments, fps):
    """[(起始帧, 帧数)] → 每段秒数。

    小数位取 5 位：加载器按 round(秒 × 源帧率) 还原帧数，30fps 下 226 帧
    写成 7.53333 秒，乘回去 225.9999 再取整还是 226，不会逐段漂。
    """
    return [round(count / fps, DURATION_DECIMALS) for _start, count in segments]



class FxAiVideoSegmenterV1:
    CATEGORY = "凤希AI/视频"
    FUNCTION = "split_segments"

    RETURN_TYPES = ("STRING", "LIST", "INT", "INT", "STRING", "AUDIO", "INT")
    RETURN_NAMES = ("视频路径", "分段时长列表", "分段总数", "帧率", "分段信息", "音频", "起始帧")

    @classmethod
    def INPUT_TYPES(cls):
        files = list_input_video_files() or [""]
        return {
            "required": {
                "视频文件": (files, {
                    "default": files[0],
                    "tooltip": "input 目录下的相对路径（子目录/文件名）。\n"
                               "面板上点「选择视频」直接挑，不用手打。\n"
                               "走 API 调用时传同样的相对路径即可，不用知道服务器物理路径。"}),
                "关键帧JSON": ("STRING", {"default": "[]", "multiline": False}),
                "跳过初始段": ("BOOLEAN", {"default": False}),
                "包含尾部段": ("BOOLEAN", {"default": True}),
                "是否平均分段": ("BOOLEAN", {"default": True}),
                "平均分段时长": ("FLOAT", {
                    "default": 10.00, "step": 0.01, "round": 0.01,
                    "tooltip": "每段多少秒。**它不改变总耗时** —— 总算力只看「总帧数 × 分辨率」，"
                               "把一段切成两段，总时间几乎一样。\n"
                               "它改变的是另外三件事：\n"
                               "  1. **第一个结果多久出来** —— 5 秒段出首片的时间是 10 秒段的一半；\n"
                               "  2. **峰值内存** —— 直接减半（超分整段都放在内存里，这段是最吃内存的）；\n"
                               "  3. **失败代价** —— 崩了只丢一段。\n"
                               "代价：段边界变多。段与段之间是硬拼（没有交叉淡化），"
                               "边界越多越可能看到一下跳变；人脸精修的滞回也会在边界处重置。\n"
                               "**建议**：长时间等待难受就调 5；追求最少接缝、机器内存够就留 10。"}),
                "目标模型": (["标准", "LTX", "MiniMax"], {
                    "default": "标准",
                    "tooltip": "标准    平均分段时长 × 帧率，四舍五入到帧。10 秒就是 10.000 秒那一帧收尾，\n"
                               "        下一段从它的下一帧开始，首尾不重叠。\n"
                               "LTX     每段帧数向下对齐到 8k+1（LTX latent 步长 8）。\n"
                               "MiniMax 每段按 H3 固定 24fps 折算成 17k+5，再换回源视频帧数。\n"
                               "三种模式的段边界都落在帧号上，循环切完不漏帧、不重复。"}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, 视频文件="", 关键帧JSON="[]", **_kwargs):
        try:
            if 视频文件:
                resolve_video_path(视频文件)
            parse_keyframe_list(关键帧JSON)
            return True
        except Exception as e:
            return str(e)

    def split_segments(self, 视频文件, 关键帧JSON="[]", 跳过初始段=False, 包含尾部段=True,
                       是否平均分段=True, 平均分段时长=10.0, 目标模型="标准"):
        path = resolve_video_path(视频文件)
        info = probe_video(path)
        fps = info["fps"]

        segments = build_frame_segments(
            info["frames"], fps, parse_keyframe_list(关键帧JSON),
            跳过初始段, 包含尾部段, 是否平均分段, 平均分段时长, 目标模型,
        )
        durations = frames_to_durations(segments, fps)

        # 掐头去尾之后的整条音轨：起点取第一段的起始帧，长度取到最后一段的尾帧。
        # 跳过初始段 / 不含尾部段时，这两头已经跟着缩掉了，下游拿到的就是
        # 真正会被切成段的那段声音，不用自己再算偏移。
        #
        # span_start 同时作为「起始帧」输出：加载器只知道每段多长，
        # 不告诉它从哪一帧起算，掐过头就会从视频第 0 帧切起，画面整体错位。
        span_start = segments[0][0]
        span_frames = segments[-1][0] + segments[-1][1] - span_start
        audio = decode_audio_range(path, span_start, span_frames, fps)

        step = segment_frames(平均分段时长, fps, 目标模型) if 是否平均分段 else 0
        head = (f"{len(segments)} 段 · {info['width']}x{info['height']} @ {format_fps(fps)}fps · "
                f"{info['frames']} 帧 · 目标模型 {目标模型}")
        if step:
            head += f" · 每段 {step} 帧（{step / fps:.3f} 秒）"
            head += "\n" + _step_detail(平均分段时长, fps, 目标模型, step)
        head += (f"\n音轨：帧 {span_start}~{span_start + span_frames - 1}"
                 f"（{span_frames} 帧 / {span_frames / fps:.3f} 秒）"
                 if audio is not None else "\n音轨：这个视频没有音频流")

        lines = [head]
        for index, (start, count) in enumerate(segments[:INFO_MAX_LINES]):
            lines.append(
                f"第{index + 1}段  帧 {start}~{start + count - 1}（{count} 帧 / "
                f"{durations[index]:.3f} 秒）"
                f"  {format_time(start / fps)} → {format_time((start + count) / fps)}"
            )
        if len(segments) > INFO_MAX_LINES:
            lines.append(f"…… 还有 {len(segments) - INFO_MAX_LINES} 段未列出")

        print(f"✅ [凤希AI] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"视频分段：{len(segments)} 段 / {info['frames']} 帧 / {目标模型}")
        # 帧率按整数往外给：下游保存节点、控制器都按整数帧率走，
        # 源素材真有 23.976 这种也四舍五入成 24（分段计算内部仍用真实值）
        return (rel_video_path(path), durations, len(segments), int(round(fps)),
                "\n".join(lines), audio, span_start)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
