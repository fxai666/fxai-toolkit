# 凤希AI：RMSNorm（自研实现）
#
# 原版（Meta Llama 2 风格）写法是：
#     x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
# 每一步都是独立 kernel，bf16 下要先把整块张量转 fp32 再规约再转回，
# DiT 36 层 × 每层 2 次，光这里就吃掉可观时间。
#
# 这里改用 torch 的原生融合算子 F.rms_norm（PyTorch 2.4+），
# 一条 kernel 完成规约 + 缩放，fp16/bf16 下都能直接算，省掉 float() 往返。
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.weight.shape[0],), self.weight, self.eps)
