import os
import re
import datetime
import torch
from fxai_audio_utils import load_wav_tensor

def list_audios(target_dir):
    if not os.path.isdir(target_dir):
        return []
    pattern = re.compile(r'(.+)\.(mp3|wav|ogg|flac|m4a)$', re.IGNORECASE)
    files = []
    for f in os.listdir(target_dir):
        fp = os.path.join(target_dir, f)
        if not os.path.isfile(fp):
            continue
        m = pattern.match(f)
        if m:
            files.append(f)
    files.sort()
    return files

def load_comfy_audio(file_path):
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return {
            "waveform": torch.zeros((1, 1, 1)),
            "sample_rate": 16000
        }
    try:
        return load_wav_tensor(file_path)
    except Exception as e:
        print(f"⚠️ 音频加载失败 {file_path}, err:{str(e)}")
        return {
            "waveform": torch.zeros((1, 1, 1)),
            "sample_rate": 16000
        }


class FxAiAudioDurationGetSeg:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "音频文件夹": ("STRING", {"default": ""}),
                "文件索引列表": ("LIST", {}),      # 上游【文件索引】[0,0,0,1,1,2...]
                "分片索引": ("INT", {"default": 0, "min": 0}), # 全局分片序号
                "分段数据": ("DICT", {})           # 上游【分段数据】
            }
        }

    RETURN_TYPES = ("AUDIO", "LIST", "INT")
    RETURN_NAMES = ("原始音频", "分段时长", "分段索引")
    FUNCTION = "run"
    CATEGORY = "凤希AI/音频"

    def run(self, 音频文件夹, 文件索引列表, 分片索引, 分段数据):
        print(f"✅ [凤希AI] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 开始渲染第 {分片索引+1} 个音频")
        target_dir = os.path.abspath(音频文件夹)
        audio_names = list_audios(target_dir)

        # 默认兜底空白音频
        out_audio = load_comfy_audio("")
        file_seg_list = []
        inner_seg_idx = 0
        raw_audio_idx = None

        # 1. 获取这条分片归属的原始音频编号
        if isinstance(文件索引列表, list):
            if 0 <= 分片索引 < len(文件索引列表):
                raw_audio_idx = 文件索引列表[分片索引]

                match_count = 0
                for i in range(分片索引 + 1):
                    if 文件索引列表[i] == raw_audio_idx:
                        match_count += 1
                inner_seg_idx = match_count - 1
            else:
                print(f"⚠️ 分片索引{分片索引}超出 文件索引列表 长度{len(文件索引列表)}")
        else:
            print("⚠️ 文件索引列表不是LIST类型！")

        # 3. 加载对应完整音频
        if raw_audio_idx is not None and isinstance(raw_audio_idx, int):
            if 0 <= raw_audio_idx < len(audio_names):
                audio_path = os.path.join(target_dir, audio_names[raw_audio_idx])
                out_audio = load_comfy_audio(audio_path)
            else:
                print(f"⚠️ 原始音频序号{raw_audio_idx}超出音频文件总数{len(audio_names)}")

            # 4. 获取该音频全部分段时长列表
            if raw_audio_idx in 分段数据:
                file_seg_list = 分段数据[raw_audio_idx]
            else:
                print(f"⚠️ 分段数据字典内不存在key={raw_audio_idx}")

        return (out_audio, file_seg_list, inner_seg_idx)