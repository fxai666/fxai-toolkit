"""把「每步只算一次」的模块放 CPU 常驻，结果回 GPU。

这类模块（H3 的 adaln_proj、token_refiner）的输入只跟时间步/文本有关，与序列长度无关：
一个 260M 参数的 adaln 每步只吃 [3, 2688] 的输入，算 11 ms，权重却要 248 MiB。
放 CPU 是**性价比最高的一种卸载** —— 卸掉 12 G 显存，只花 0.58 s/步，还能藏掉。

三种用法：
    m = CpuModule(block.adaln_proj, "cuda:0")     # 权重搬到 CPU，显存立刻释放
    outs = m(t_emb)                               # 同步，返回已在 GPU 上的结果
    fut = m.submit(t_emb_next)                    # 后台算下一步
    outs = fut.result()                           # 到用时再取，通常已经算完
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import torch

__all__ = ["CpuModule", "move_once_modules_to_cpu"]


def _dequant_to(module, dtype):
    """把量化权重**真正**反量化成 dtype，返回 (module, 反量化了几份)。

    **不能靠 `module.to(dtype)`** —— `QuantizedTensor._handle_to` 只改
    `orig_dtype` 元数据、存储仍是 int8：`.to(bf16)` 之后跑的还是 int8 kernel
    （实测慢 10 倍，110 ms/层 vs 10.7 ms/层），而 `element_size()` 又按逻辑
    dtype 报，体积还多算一倍。必须走 `.dequantize()`。
    """
    n = [0]

    def cast(t):
        if hasattr(t, "dequantize"):
            t = t.dequantize()
            n[0] += 1
        return t.to(dtype) if t.dtype != dtype else t

    module._apply(cast)
    return module, n[0]


class CpuModule:
    """把一个 nn.Module 常驻 CPU，调用时自动把结果送回 device。"""

    def __init__(self, module, device, dtype=torch.bfloat16, name=""):
        self.device = torch.device(device)
        self.dtype = dtype
        self.name = name or type(module).__name__
        self.module = module.to("cpu")
        self.n_dequant = 0
        if dtype is not None:
            self.module, self.n_dequant = _dequant_to(self.module, dtype)
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.module.eval()
        self._pool = ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix=f"cpu-once-{self.name}")
        self._lock = threading.Lock()
        self._bytes = sum(t.numel() * t.element_size() for t in self.module.parameters())

    @property
    def ram_gib(self) -> float:
        return self._bytes / (1 << 30)

    # ---------------------------------------------------------------- 内部
    def _to_device(self, out, device=None):
        dev = torch.device(device) if device is not None else self.device
        if isinstance(out, tuple):
            return tuple(o.to(dev, non_blocking=True) if torch.is_tensor(o) else o
                         for o in out)
        if isinstance(out, list):
            return [o.to(dev, non_blocking=True) if torch.is_tensor(o) else o
                    for o in out]
        return out.to(dev, non_blocking=True) if torch.is_tensor(out) else out

    # ---------------------------------------------------------------- 对外
    def compute_cpu(self, x):
        """在 CPU 上算，结果**留在 CPU**。多卡共用同一份 modulation 时用这个 ——
        算一次，各 rank 自己搬回本卡（结果只有几百 KB，搬运可忽略）。"""
        with self._lock, torch.inference_mode():
            return self.module(x.detach().to("cpu"))

    def __call__(self, x, device=None):
        return self._to_device(self.compute_cpu(x), device)

    def submit(self, x, device=None):
        """后台算，返回 future。x 必须先 detach 并留在原地（别让调用方改它）。"""
        return self._pool.submit(self.compute_cpu, x)

    def release(self):
        self._pool.shutdown(wait=False)
        self.module = None

    def __repr__(self):
        return f"<CpuModule {self.name} {self.ram_gib:.2f}G RAM → {self.device}>"


def move_once_modules_to_cpu(blocks, names=("adaln_proj",), device="cuda:0",
                             parallel_blocks=None):
    """按名字把每个 block 上的子模块换成 CpuModule，返回替换表。

    显存是**立刻**释放的（nn.Module.to 是原地搬运），不需要等 GC。

    `ParallelDiTBlock` 在构造时就把 `block.adaln_proj` 存成了自己的属性，所以如果你
    已经建好了并行 block，必须把 `parallel_blocks` 一起传进来，否则它手里还是旧引用。
    """
    table = {}
    for i, blk in enumerate(blocks):
        for nm in names:
            mod = getattr(blk, nm, None)
            if mod is None or isinstance(mod, CpuModule):
                continue
            key = f"{i}.{nm}"
            wrap = CpuModule(mod, device, name=f"blk{i}.{nm}")
            table[key] = wrap
            setattr(blk, nm, wrap)
            if parallel_blocks is not None and i < len(parallel_blocks):
                setattr(parallel_blocks[i], nm, wrap)
    return table
