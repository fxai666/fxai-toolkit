# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# 音频列表预览：把 AUDIO 列表里的每段音频写成 wav 临时文件，
# 通过 ui.audio 数据返回给前端，前端按列表渲染多个播放器。

import os
import time

import soundfile as sf

import folder_paths


def _save_audio_to_temp(audio, index):
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])

    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform.reshape(1, -1)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    audio_np = waveform.cpu().numpy().T
    if audio_np.shape[1] > 2:
        audio_np = audio_np[:, :2]

    filename = f"fxai_audio_list_{int(time.time())}_{index:03d}.wav"
    full_path = os.path.join(folder_paths.get_temp_directory(), filename)
    sf.write(full_path, audio_np, sample_rate)
    return filename


class FxAiAudioListPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "音频": ("LIST", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("音频数量",)
    FUNCTION = "preview"
    CATEGORY = "凤希AI/音频"
    OUTPUT_NODE = True

    def preview(self, 音频):
        results = []
        for index, audio in enumerate(音频 or []):
            if not isinstance(audio, dict) or "waveform" not in audio:
                continue
            try:
                filename = _save_audio_to_temp(audio, index)
                results.append({"filename": filename, "subfolder": "", "type": "temp"})
            except Exception as e:
                print(f"[凤希AI音频列表预览] 第{index + 1}段保存失败: {e}")

        return {"ui": {"audio": results}, "result": (len(results),)}
