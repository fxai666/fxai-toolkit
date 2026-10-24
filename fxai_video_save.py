# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# 通用视频保存 V1：帧序列 → mp4，不绑任何模型。
# 修脸、补帧、超分这类「帧原样走一遍」的流程用它，不占 MiniMax 那套的口径。
#
# 与 MiniMax 视频保存 V2 的差别只有音频可选：没接就出无声视频，不再因为缺音轨卡住。
# 帧率默认 24，可直接接拆帧节点的帧率输出；编码与入库复用 V2 的 save_video，
# 不复制第二份 ffmpeg 代码。

import gc
import os

import torch

import fxai_task_store
from fxai_minimax_video_save_v2 import get_video_dir, save_video


class FxAiVideoSaveV1:
    CATEGORY = "凤希AI/视频"
    FUNCTION = "run"

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("过渡帧", "视频文件路径", "保存目录")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片序列": ("IMAGE",),
                "视频序号": ("INT", {"default": -1, "min": -1}),
                "保存目录": ("STRING", {"default": "sucai"}),
                "帧率FPS": ("INT", {"default": 24, "min": 1, "max": 240, "step": 1}),
            },
            "optional": {
                "音频": ("AUDIO",),
            },
        }

    def run(self, 图片序列, 视频序号=-1, 保存目录="sucai", 帧率FPS=24, 音频=None):
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
