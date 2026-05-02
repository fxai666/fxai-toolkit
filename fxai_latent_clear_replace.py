import torch
import copy

class FxAiLatentClearReplace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "潜空间序列": ("LATENT",),
                "letter": ("LATENT",),      # 同格式 LTXV 潜空间
                "帧索引": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "凤希AI/工具"

    def run(self, 潜空间序列, letter, 帧索引):
        out = copy.deepcopy(潜空间序列)
        latent = out["samples"]
        B, C, total_frames, H, W = latent.shape

        # 安全索引
        if 帧索引 < 0:
            idx = total_frames + 帧索引
        else:
            idx = 帧索引
        idx = max(0, min(idx, total_frames - 1))

        # 取 letter 的第一帧
        letter_data = letter["samples"]
        source = letter_data[:, :, :1, :, :]

        # 直接替换（同格式才能成功！）
        latent[:, :, idx:idx+1, :, :] = source

        return (out,)