# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# MiniMax H3 块缓存加速节点（block_loop 钩子由 fxai_minimax_core_patch 注入，
# 这里只负责按 denoise 步选择 FULL/CACHE 执行，不修改任何系统文件）。
#
# 原理：FULL 步跑全部 transformer 块，并用 ("double_block", i) 钩子测量每个块
# 的更新幅度（|h_out - h_in|），同时在当前缓存边界处捕获快照，残差 = 全块输出
# - 快照。CACHE 步只重算前 k 个 warm 块，加上残差近似尾部块贡献。
#
# 自适应块选择：跳过哪些块不固定为尾部比例，而是按 FULL 步实测的各块更新幅度
# 决定——更新幅度小的块优先跳过，跳过的累计贡献不超过 cache_depth 预算。这样
# 静态层多的模型可以跳更多，活跃层多的模型自动保守。在 denoise 中间窗口且
# sigma 变化小、连续缓存步未超限时允许使用缓存，避免误差累积。

import torch


class _MiniMaxH3Cache:
    """按 denoise 步在 FULL 与 CACHE 之间切换的块缓存控制。"""

    def __init__(self, control_value, start_percent, end_percent, mcs, device, cache_depth=0.75):
        self.threshold = float(control_value)
        self.start_percent = float(start_percent)
        self.end_percent = float(end_percent)
        self.mcs = int(mcs)
        self.cache_device = str(device)
        self.cache_depth = float(cache_depth)
        self.enabled = (self.threshold > 0.0) and (self.mcs > 0) and (self.cache_depth > 0.0)
        self.preset_name = "均衡"
        self.reset()

    def reset(self):
        self.start_sigma = None
        self.last_sigma = None
        self.prev_sigma = None
        self.step = -1
        self.total_steps = None
        self.consecutive_skips = 0
        self.residual = None
        self.snapshot = None
        self.sigma_scale = 1.0
        self.last_mode = "full"
        self.full_steps = 0
        self.cache_hits = 0
        self.skipped_blocks = 0
        self.total_blocks = 0
        self.printed = False
        self.measuring = False
        self.block_count = 0
        self.boundary = 0
        self.block_deltas = None
        self.measure_counts = None

    def __call__(self, args, kwargs):
        """("block_loop", 0) 钩子入口。args 为打包的隐藏态，kwargs 含原块循环。"""
        original_block = kwargs["original_block"]
        block_count = int(args["block_count"])
        sigma = self._current_sigma(args)
        if sigma is None:
            return {"img": original_block(args)["img"]}

        if self.last_sigma is None or sigma > self.last_sigma + 1e-6:
            self._finish_run()
            self.reset()
            self.block_count = block_count
            self.boundary = self._warm_blocks(block_count)
            self.start_sigma = sigma
            self.sigma_scale = self._detect_scale(args, sigma)
            self.last_sigma = sigma
            self.step = 0
            self.last_mode = "full"
            return {"img": self._full_step(args, kwargs, block_count, count=True)}

        if abs(sigma - self.last_sigma) <= 1e-6:
            if self.last_mode == "cache":
                return {"img": self._cache_step(args, kwargs, block_count, count=False)}
            return {"img": self._full_step(args, kwargs, block_count, count=False)}

        self.prev_sigma = self.last_sigma
        self.last_sigma = sigma
        self.step += 1
        sigma_n = sigma / self.sigma_scale
        prev_n = self.prev_sigma / self.sigma_scale
        pos = self._position(args, sigma_n)
        if self.total_steps is not None and self.step >= self.total_steps:
            self._print_stats()
        k = self.boundary
        in_window = self.start_percent <= pos <= self.end_percent
        slow = abs(prev_n - sigma_n) < self.threshold
        can_skip = (self.enabled and k < block_count and self.residual is not None
                    and in_window and slow and self.consecutive_skips < self.mcs)
        if can_skip:
            self.last_mode = "cache"
            return {"img": self._cache_step(args, kwargs, block_count, count=True)}
        self.last_mode = "full"
        return {"img": self._full_step(args, kwargs, block_count, count=True)}

    def _warm_blocks(self, block_count):
        """无测量时的默认 warm 块数，与官方固定尾部缓存一致。"""
        return max(0, min(block_count - 1, round(block_count * (1.0 - self.cache_depth))))

    def _adapt_boundary(self):
        """按实测各块更新幅度选择缓存边界：从尾部累加，跳过的累计贡献不超过预算。"""
        block_count = self.block_count
        if self.block_deltas is None or self.measure_counts is None:
            self.boundary = self._warm_blocks(block_count)
            return
        avg = [self.block_deltas[i] / max(1, self.measure_counts[i]) for i in range(block_count)]
        total = sum(avg)
        if total <= 0.0:
            self.boundary = self._warm_blocks(block_count)
            return
        budget = self.cache_depth * total
        acc = 0.0
        k = block_count
        while k > 1:
            if acc + avg[k - 1] > budget:
                break
            acc += avg[k - 1]
            k -= 1
        self.boundary = k

    def measure(self, index, h_in, out):
        """FULL 步逐块累计更新幅度，采样一半隐藏维以降低测量开销。"""
        if self.block_deltas is None or len(self.block_deltas) != self.block_count:
            self.block_deltas = [0.0] * self.block_count
            self.measure_counts = [0] * self.block_count
        d = (out[:, ::16] - h_in[:, ::16]).abs().mean().item()
        self.block_deltas[index] += d
        self.measure_counts[index] += 1

    def _full_step(self, args, kwargs, block_count, count=True):
        if count:
            self.full_steps += 1
            self.consecutive_skips = 0
            self.total_blocks += block_count
        self.measuring = True
        self._adapt_boundary()
        original_block = kwargs["original_block"]
        h = original_block(args)["img"]
        self.measuring = False
        if self.snapshot is not None:
            residual = h - self.snapshot
        else:
            residual = h - args["img"].clone()
        self.residual = residual.to("cpu") if self.cache_device == "cpu" else residual
        return h

    def _cache_step(self, args, kwargs, block_count, count=True):
        if count:
            self.cache_hits += 1
            self.consecutive_skips += 1
            self.total_blocks += block_count
            self.skipped_blocks += block_count - self.boundary
        original_block = kwargs["original_block"]
        h = args["img"]
        k = self.boundary
        if k > 0:
            h = original_block({**args, "start": 0, "end": k})["img"]
        if self.residual is not None:
            h = h + self.residual.to(h.device)
        return h

    @staticmethod
    def _current_sigma(args):
        timestep = args["transformer_options"].get("sigmas")
        if timestep is None:
            return None
        return float(torch.as_tensor(timestep).flatten()[0].float())

    def _detect_scale(self, args, sigma):
        sample_sigmas = args["transformer_options"].get("sample_sigmas")
        if sample_sigmas is not None:
            ss0 = float(torch.as_tensor(sample_sigmas).flatten().float()[0])
            if abs(ss0) > 1e-9:
                s = sigma / ss0
                if abs(s) > 1e-9:
                    return s
        return 1.0

    def _position(self, args, sigma_n):
        sample_sigmas = args["transformer_options"].get("sample_sigmas")
        if sample_sigmas is not None:
            ss = torch.as_tensor(sample_sigmas).flatten().float()
            if ss.numel() > 1:
                self.total_steps = ss.numel() - 1
                idx = int((ss - sigma_n).abs().argmin())
                return min(1.0, max(0.0, idx / self.total_steps))
        if self.start_sigma is not None and self.start_sigma > 0.0:
            start_n = self.start_sigma / self.sigma_scale
            return min(1.0, max(0.0, (start_n - sigma_n) / start_n))
        return 1.0

    def _finish_run(self):
        if not self.printed and (self.full_steps + self.cache_hits) > 0:
            self._print_stats()

    def _print_stats(self):
        self.printed = True
        if self.total_blocks <= 0:
            return
        saved = 100.0 * self.skipped_blocks / self.total_blocks
        print(f"【凤希AI】加速生效，预计加速 {saved:.0f}% （{self.preset_name}档）。如需调整请切换节点的【速度档位】。")


class _MiniMaxH3BlockHook:
    """("double_block", i) 钩子：FULL 步测量各块更新幅度，并在缓存边界捕获快照。"""

    def __init__(self, cache, index):
        self.cache = cache
        self.index = index

    def __call__(self, args, kwargs):
        h_in = args["img"]
        out = kwargs["original_block"](args)
        if self.cache.measuring:
            self.cache.measure(self.index, h_in, out["img"])
        if self.index == self.cache.boundary:
            self.cache.snapshot = h_in.clone()
        return out


# 档位基准参数：sigma阈值 / 窗口起点 / 窗口终点 / 最大连续缓存步数 / 缓存深度
_PRESETS = {
    "不加速": (0.0, 0.0, 1.0, 0, 0.0),
    "画质优先": (0.08, 0.15, 0.85, 1, 0.5),
    "均衡": (0.12, 0.10, 0.90, 2, 0.75),
    "极速": (0.16, 0.05, 0.95, 3, 0.85),
}


def _resolve_params(档位, 采样步数):
    sigma阈值, 起点, 终点, mcs, 深度 = _PRESETS[档位]
    steps = int(采样步数)
    if steps < 12:
        sigma阈值 *= 0.7
        深度 = min(0.45, 深度 * 0.6)
        mcs = min(mcs, 1)
    elif steps >= 20:
        深度 = min(0.95, 深度 * 1.1)
    return sigma阈值, 起点, 终点, mcs, 深度


class FxAiMiniMaxBlockCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": ("MODEL",),
                "速度档位": (["不加速", "画质优先", "均衡", "极速"], {
                    "default": "均衡",
                    "tooltip": "加速档位：均衡为默认；画质优先更保守；极速更快但误差稍大；不加速为官方原速。"}),
                "采样步数": ("INT", {
                    "default": 20, "min": 8, "max": 100, "step": 1,
                    "tooltip": "本次生成的采样步数，会输出给采样器使用。步数越少内部自动越保守。"}),
            },
        }

    RETURN_TYPES = ("MODEL", "INT")
    RETURN_NAMES = ("模型", "采样步数")
    FUNCTION = "patch"
    CATEGORY = "凤希AI/MiniMax"

    def patch(self, 模型, 速度档位, 采样步数):
        inner = self._find_minimax_dit(模型)
        if inner is None:
            raise ValueError("FxAiMiniMaxBlockCache 仅支持 MiniMax H3 模型")
        sigma阈值, 缓存窗口起点, 缓存窗口终点, 最大连续缓存步数, 缓存深度 = _resolve_params(速度档位, 采样步数)
        cache = _MiniMaxH3Cache(sigma阈值, 缓存窗口起点,
                                缓存窗口终点, 最大连续缓存步数, "auto", 缓存深度)
        cache.preset_name = 速度档位
        model = 模型.clone()
        if not cache.enabled:
            return (model, int(采样步数))
        model.set_model_patch_replace(cache, "dit", "block_loop", 0)
        block_count = len(inner.blocks)
        cache.block_count = block_count
        cache.boundary = cache._warm_blocks(block_count)
        for i in range(block_count):
            model.set_model_patch_replace(_MiniMaxH3BlockHook(cache, i), "dit", "double_block", i)
        return (model, int(采样步数))

    @staticmethod
    def _find_minimax_dit(model):
        m = getattr(model, "model", None)
        seen = 0
        while m is not None and seen < 12:
            if type(m).__name__ == "MiniMaxH3Model":
                return m
            nxt = None
            for attr in ("model", "inner_model", "diffusion_model", "unet_model"):
                nxt = getattr(m, attr, None)
                if nxt is not None:
                    break
            m = nxt
            seen += 1
        return None

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")