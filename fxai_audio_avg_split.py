import torch
import numpy as np

def _normalize_audio_tensor(音频):
    waveform = 音频["waveform"]
    sample_rate = 音频["sample_rate"]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    return waveform, sample_rate

def _build_average_segments(total_duration, average_duration):
    total_duration = max(0.0, total_duration)
    average_duration = max(0.1, average_duration)
    
    segments = []
    current = 0.0
    while current < total_duration:
        end = current + average_duration
        if end > total_duration:
            end = total_duration
        segments.append((current, end))
        current = end

    min_threshold = 5.0
    if len(segments) >= 2:
        last_start, last_end = segments[-1]
        last_length = last_end - last_start
        if last_length < min_threshold:
            segments.pop()
            prev_start, _ = segments[-1]
            segments[-1] = (prev_start, last_end)

    return segments

class FxAiAudioAvgSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "音频": ("AUDIO", {"forceInput": True}),
                "平均分段时长": ("FLOAT", {"default": 15.00, "step": 0.01, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("分段列表",)
    FUNCTION = "run"
    CATEGORY = "凤希AI/音频"

    def run(self, 音频, 平均分段时长):
        waveform, sample_rate = _normalize_audio_tensor(音频)
        total_duration = waveform.shape[-1] / sample_rate
        segments = _build_average_segments(total_duration, 平均分段时长)
        duration_list = [round(e - s, 2) for s, e in segments]
        return (duration_list,)