# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# 通用视频保存 V3：全部帧写入视频，帧率可填也可接、音频可选。
#
# 与 V2 的差别只有两处，都是为了给「帧序列原样走一遍」的流程用
# （修脸、补帧、超分这类），V2 本身一行不动，老工作流不受影响：
#   1. 音频 从必填改成可选 —— 没接就输出无声视频，不再因为缺音轨卡住；
#   2. 帧率 从写死 24 改成面板参数，且是 FLOAT，可以直接接拆帧节点的 帧率。
#
# 编码、入库、广播仍然复用 V2 的 save_video，不复制第二份 ffmpeg 代码。

import gc
import os

import torch

import fxai_task_store
from fxai_minimax_video_save_v2 import get_video_dir, save_video


class FxAiMiniMaxVideoSaveV3:
    CATEGORY = "凤希AI/视频"
    FUNCTION = "run"

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("过渡帧", "视频文件路径", "保存目录")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片序列": ("IMAGE",),
                "保存目录": ("STRING", {"default": "sucai",
                    "tooltip": "fxai/video/ 下的子目录名，不存在会自动创建。\n"
                               "修脸、补帧这类独立处理建议单独开一个目录，别和成片混在一起。"}),
                "帧率FPS": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 1.0,
                    "tooltip": "输出视频帧率。可以直接接拆帧节点的「帧率」输出——\n"
                               "修脸这种帧数不变、原样走一遍的流程，接了就不用再手填。"}),
                "视频序号": ("INT", {"default": -1, "min": -1,
                    "tooltip": "-1 = 时间戳命名（fxai_年月日_时分秒.mp4），不会覆盖已有文件；\n"
                               "填 0 及以上的整数 = 按 000.mp4 / 001.mp4 命名，供多段流程按序号拼接。"}),
            },
            "optional": {
                "音频": ("AUDIO", {
                    "tooltip": "可选。接上就把这条音轨封装进 mp4（AAC 192k，取较短的一边）；\n"
                               "不接则输出无声视频。音频节点（音频加载 / 音频管理）都能直接接进来。"}),
            },
        }

    def run(self, 图片序列, 保存目录="sucai", 帧率FPS=24.0, 视频序号=-1, 音频=None):
        if 图片序列 is None:
            return (图片序列, "", "")

        save_subdir = (保存目录 or "sucai").strip() or "sucai"
        target_dir = get_video_dir(save_subdir)

        过渡帧 = 图片序列[-1:]

        video_path = save_video(
            images=图片序列,
            save_dir=target_dir,
            audio=音频,
            fps=帧率FPS,
            custom_num=视频序号,
        )

        del 图片序列
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if video_path and os.path.exists(video_path):
            try:
                if 视频序号 < 0:
                    fxai_task_store.save_result("video", save_subdir,
                                                [os.path.basename(video_path)])
                else:
                    fxai_task_store.broadcast("scene_saved", {
                        "scene_index": 视频序号,
                        "scene_count": 视频序号 + 1,
                        "path": video_path,
                        "message": f"第 {视频序号 + 1} 个场景视频已生成",
                    })
            except Exception as e:
                print(f"[凤希AI] 视频广播失败：{e}")

        return (过渡帧, video_path, target_dir)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
