import os
import re
import subprocess
import json

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

def get_audio_duration(audio_path):
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            audio_path
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10
        )
        meta = json.loads(result.stdout)
        duration = float(meta["format"]["duration"])
        return round(duration, 3)
    except Exception:
        return 0.0

def split_duration_segments(orig_dur, split_gap=15.0, split_trigger=20.0):
    """
    时长分段算法
    orig_dur: 原始音频时长
    split_gap: 分片长度
    split_trigger: 超过该值才进行分片（20秒）
    return: list 片段时长数组
    """
    segments = []
    if orig_dur <= split_trigger:
        segments.append(round(orig_dur,3))
        return segments

    remain = orig_dur
    while remain > 0:
        if remain > split_gap:
            segments.append(split_gap)
            remain -= split_gap
        else:
            segments.append(round(remain,3))
            remain = 0
    return segments


class FxAiAudioDurationSeg:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文件夹路径": ("STRING", {"default": ""}),
                "分片秒数": ("FLOAT", {"default":15.0, "min":1.0, "step":0.5})
            }
        }

    # 修复：输出3个通道，必须对应3个类型
    RETURN_TYPES = ("LIST", "LIST", "DICT")
    RETURN_NAMES = ("分段时长","文件索引","分段数据")
    FUNCTION = "run"
    CATEGORY = "凤希AI/音频"

    def run(self, 文件夹路径, 分片秒数):
        target_dir = os.path.abspath(文件夹路径)
        file_names = list_audios(target_dir)

        segment_list = []
        file_list = []
        segment_data = {}

        for file_idx, filename in enumerate(file_names):
            audio_path = os.path.join(target_dir, filename)
            dur = get_audio_duration(audio_path)
            seg_durations = split_duration_segments(dur, split_gap=分片秒数, split_trigger=20.0)
            segment_data[file_idx] = []
            for seg in seg_durations:
                segment_list.append(seg)
                file_list.append(file_idx)
                segment_data[file_idx].append(seg)

        return (segment_list, file_list, segment_data)