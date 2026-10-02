# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权

import torch
import torch.nn.functional as F
import torchaudio.functional as TAF

import folder_paths

from fxai_melroformer_arch import MelBandRoformer

from comfy import model_management as mm
from comfy.utils import load_torch_file, ProgressBar

_MODEL_CONFIG = {
    "dim": 384,
    "depth": 6,
    "stereo": True,
    "num_stems": 1,
    "time_transformer_depth": 1,
    "freq_transformer_depth": 1,
    "num_bands": 60,
    "dim_head": 64,
    "heads": 8,
    "attn_dropout": 0,
    "ff_dropout": 0,
    "flash_attn": True,
    "dim_freqs_in": 1025,
    "sample_rate": 44100,
    "stft_n_fft": 2048,
    "stft_hop_length": 441,
    "stft_win_length": 2048,
    "stft_normalized": False,
    "mask_estimator_depth": 2,
    "multi_stft_resolution_loss_weight": 1.0,
    "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256),
    "multi_stft_hop_size": 147,
    "multi_stft_normalized": False,
}


def _get_windowing_array(window_size, fade_size, device):
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window.to(device)


class FxAiMelRoformerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型文件": (folder_paths.get_filename_list("diffusion_models"),),
            }
        }

    RETURN_TYPES = ("MELROFORMERMODEL",)
    RETURN_NAMES = ("模型",)
    FUNCTION = "load"
    CATEGORY = "凤希AI/音频"

    def load(self, 模型文件):
        print(f"✅ [凤希AI] 加载人声分离模型：{模型文件}")
        model = MelBandRoformer(**_MODEL_CONFIG).eval()
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", 模型文件)
        model.load_state_dict(load_torch_file(model_path), strict=True)
        return (model,)


class FxAiMelRoformerSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": ("MELROFORMERMODEL",),
                "音频": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("人声", "伴奏")
    FUNCTION = "process"
    CATEGORY = "凤希AI/音频"

    def process(self, 模型, 音频):
        print("✅ [凤希AI] 音频即将进行人声分离")
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        audio_input = 音频["waveform"]
        sample_rate = 音频["sample_rate"]
        _batch, audio_channels, audio_length = audio_input.shape

        sr = 44100

        if audio_channels == 1:
            audio_input = audio_input.repeat(1, 2, 1)
            audio_channels = 2

        if sample_rate != sr:
            print(f"✅ [凤希AI] 重采样 {sample_rate} → {sr}")
            audio_input = TAF.resample(audio_input, orig_freq=sample_rate, new_freq=sr)
        audio_input = original_audio = audio_input[0]

        C = 352800
        N = 2
        step = C // N
        fade_size = C // 10
        border = C - step

        if audio_length > 2 * border and border > 0:
            audio_input = F.pad(audio_input, (border, border), mode='reflect')

        windowing_array = _get_windowing_array(C, fade_size, device)

        audio_input = audio_input.to(device)
        vocals = torch.zeros_like(audio_input, dtype=torch.float32).to(device)
        counter = torch.zeros_like(audio_input, dtype=torch.float32).to(device)

        total_length = audio_input.shape[1]
        num_chunks = (total_length + step - 1) // step

        模型.to(device)

        comfy_pbar = ProgressBar(num_chunks)

        for i in range(0, total_length, step):
            part = audio_input[:, i:i + C]
            length = part.shape[-1]
            if length < C:
                if length > C // 2 + 1:
                    part = F.pad(input=part, pad=(0, C - length), mode='reflect')
                else:
                    part = F.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)

            x = 模型(part.unsqueeze(0))[0]

            window = windowing_array.clone()
            if i == 0:
                window[:fade_size] = 1
            elif i + C >= total_length:
                window[-fade_size:] = 1

            vocals[..., i:i+length] += x[..., :length] * window[..., :length]
            counter[..., i:i+length] += window[..., :length]
            comfy_pbar.update(1)

        模型.to(offload_device)

        estimated_sources = vocals / counter

        if audio_length > 2 * border and border > 0:
            estimated_sources = estimated_sources[..., border:-border]

        vocals_out = {
            "waveform": estimated_sources.unsqueeze(0).cpu(),
            "sample_rate": sr,
        }
        instruments_out = {
            "waveform": (original_audio.to(device) - estimated_sources).unsqueeze(0).cpu(),
            "sample_rate": sr,
        }

        print("✅ [凤希AI] 人声分离完成")
        return (vocals_out, instruments_out)
