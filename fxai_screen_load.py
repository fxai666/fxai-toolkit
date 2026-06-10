import os
import glob
import numpy as np
from pydub import AudioSegment

class FxAiScreenLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "场景数据": ("LIST", {"forceInput": True}),
                "行索引": ("INT", {"forceInput": True}),
            },
            "optional": {
                "音频目录": ("STRING", {"forceInput": True}),
                "通用提示词": ("STRING", {"default": "", "forceInput": True}),
                "尾部通用提示词": ("STRING", {"default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING","BOOLEAN", "AUDIO")
    RETURN_NAMES = ("提示词","图片地址","启用转场","截取音频")

    FUNCTION = "get_scene_data"
    CATEGORY = "凤希AI/影视剧场"

    def get_scene_data(self, 场景数据, 行索引, 音频目录="", 通用提示词="", 尾部通用提示词=""):
        try:
            total_lines = len(场景数据) if isinstance(场景数据, list) else 0

            if 行索引 < 0 or total_lines == 0 or 行索引 >= total_lines:
                raise IndexError(f"行索引越界：{行索引} / 总行数：{total_lines}")

            item = 场景数据[行索引]
            提示词 = 通用提示词 + item["提示词文本"] + 尾部通用提示词
            音频开始 = float(item["音频开始"])
            时长 = float(item["时长"])
            音频索引 = int(item["音频索引"])
            图片地址 = int(item["图片地址"])
            启用转场 = int(item["转场"]) == 1

            # 初始化直接 = None
            截取音频 = None

            # 索引不是 -1 才处理
            if 音频索引 > -1 and 音频目录:
                # 支持格式
                audio_exts = ('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.wma')
                
                audio_files = []
                for ext in audio_exts:
                    audio_files.extend(glob.glob(os.path.join(音频目录, f"*{ext}")))
                
                # 自动排序
                audio_files = sorted(audio_files)
                audio_files = [os.path.basename(f) for f in audio_files]

                # 取索引对应音频
                if 音频索引 < len(audio_files):
                    file_path = os.path.join(音频目录, audio_files[音频索引])
                    audio = AudioSegment.from_file(file_path)

                    start_ms = int(音频开始 * 1000)
                    duration_ms = int(时长 * 1000)
                    cut_audio = audio[start_ms : start_ms + duration_ms]

                    截取音频 = self.to_comfy_audio(cut_audio)

            return (
                提示词,
                图片地址,
                启用转场,
                截取音频
            )

        except Exception as e:
            print(f"✅ [凤希AI场景加载] 异常：{e}")
            return (
                f"{通用提示词}{尾部通用提示词}",
                图片地址,
                True,
                None
            )

    def to_comfy_audio(self, segment):
        sample_rate = segment.frame_rate
        channels = segment.channels
        audio_data = np.frombuffer(segment.raw_data, dtype=np.int16).astype(np.float32) / 32767.0
        audio_data = audio_data.reshape((-1, channels))
        return (audio_data, sample_rate)