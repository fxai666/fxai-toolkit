# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
"""把已加载的 `MiniMaxH3Model` 变成多卡张量并行版本。

做法只有三件事，全部用**代理**完成，不改 ComfyUI 源码：

1. `model.blocks[i]` 换成 `RankBlockProxy` —— 里面按 rank 各存一份
   `ParallelDiTBlock`，调用时按当前线程的 rank 取。
2. 不切分的模块（patch_proj / condition_proj / token_refiner / final_layer …）
   每卡复制一份，同样用代理分发。它们必须每卡一份：`_forward` 里这些模块吃的
   是本卡张量，权重留在别的卡上会直接 device mismatch。
3. `model._forward` 换成「起 world 个线程各跑一遍原 `_forward`」的包装。
   原 `_forward` 一行不改 —— 布局、mod_segments、adaln 曲线、final_layer 全都
   沿用 ComfyUI 的代码，ComfyUI 升级模型结构时不会失效。

四个必须处理的坑
----------------
- **`prefetch_dynamic_vbars` 必须摘掉**。开了 `--enable-dynamic-vram` 时
  `make_prefetch_queue` 会为每个 block 建预取队列，而我们的代理 block 不在它的
  账本里，预取会去搬不存在的权重。
- **norm / q_norm / adaln 必须每卡 deepcopy**。`ParallelDiTBlock` 直接引用原
  block 的子模块，两个 rank 拿到的是同一个对象；`rank1` 一 `.to("cuda:1")` 就把
  rank0 的模块搬走了。
- **adaln 卸到 CPU 时必须每步只算一次**。50 个 block 各调一次就是 100 次 CPU
  前向（两卡），实测 10.7 ms/层 → 1.07 s/步，白扔。按 `t_emb` 的对象身份缓存，
  一个 step 内所有 block 共用一份 modulation。
- **`token_refiner` 卸 CPU 后必须缓存输出**。它的输入只依赖文本条件、跨步不变；
  实测 S_text=512 时 CPU 单次 4.0 s，20 步就是 80 s，缓存之后只花 4 s（一次）。
"""
from __future__ import annotations

import copy
import threading

import torch

from .block import ParallelDiTBlock, check_split
from .exchange import HostExchange
from .offload import CpuModule

__all__ = ["H3ParallelRunner", "install_parallel"]

# 必须每卡一份的非 block 模块（体积都小，加起来 < 1.6G）
NON_BLOCK = ("video_patch_proj", "audio_patch_proj", "condition_proj",
             "token_refiner", "time_embedder", "final_layer")

# 这些名字对应 planner 里的卸载组
ONCE_MODULES = {"token_refiner": "refiner", "time_embedder": "cond"}


class _Memo:
    """按输入张量的对象身份记忆结果。

    同一个 step 内 `t_emb` / 文本条件是同一个张量对象，跨 step 才换新对象。
    强引用旧输入，避免 id 被复用导致误命中。
    """

    __slots__ = ("_key", "_keep", "_out", "_lock")

    def __init__(self):
        self._key = None
        self._keep = None
        self._out = None
        self._lock = threading.Lock()

    @staticmethod
    def key_of(x):
        try:
            ver = x._version          # inference tensor 没有版本计数器
        except RuntimeError:
            ver = None
        return (id(x), ver, tuple(x.shape), x.dtype)

    def get(self, key):
        with self._lock:
            return self._out if self._key == key else None

    def put(self, key, x, out):
        with self._lock:
            self._key, self._keep, self._out = key, x, out
        return out


class RankModuleProxy:
    """按当前线程的 rank 分发到一个普通 nn.Module。"""

    def __init__(self, per_rank, exchange, name=""):
        self.per_rank = per_rank
        self.ex = exchange
        self.name = name

    def __call__(self, *a, **kw):
        return self.per_rank[self.ex.rank](*a, **kw)

    def __repr__(self):
        return f"<RankModuleProxy {self.name} world={len(self.per_rank)}>"


def _set_plain(model, name, value):
    """把代理挂到 nn.Module 上。

    `nn.Module.__setattr__` 只接受 Module / Parameter，普通对象会被拒。所以要先把
    名字从 `_modules` 里摘掉，再写进实例 `__dict__` —— 这样 `model.name` 走的是
    普通属性查找，`state_dict()` 里也不再出现它。
    """
    model._modules.pop(name, None)
    model.__dict__[name] = value


class CpuOnceProxy:
    """「每步只算一次」的模块：CPU 上算一次、结果缓存，各 rank 自己搬回本卡。

    输入跨 step 不变（文本条件），所以第一次算完之后后面全是缓存命中。
    即使两个 rank 同时进来，也只会有一次真正的 CPU 计算（单飞锁）。
    """

    def __init__(self, cpu_mod, exchange, devices, name=""):
        self.cpu_mod = cpu_mod
        self.ex = exchange
        self.devices = devices
        self.name = name
        self._memo = _Memo()
        self._flight = threading.Lock()

    def __call__(self, x, *a, **kw):
        key = _Memo.key_of(x)
        out = self._memo.get(key)
        if out is None:
            with self._flight:
                out = self._memo.get(key)
                if out is None:
                    with torch.inference_mode():
                        out = self.cpu_mod.module(x.detach().to("cpu"), *a, **kw)
                    self._memo.put(key, x, out)
        dev = self.devices[self.ex.rank]
        if isinstance(out, (tuple, list)):
            return type(out)(o.to(dev, non_blocking=True) if torch.is_tensor(o) else o
                             for o in out)
        return out.to(dev, non_blocking=True) if torch.is_tensor(out) else out

    def __repr__(self):
        return f"<CpuOnceProxy {self.name}>"


class RankBlockProxy:
    """`model.blocks[i]` 的替身：按 rank 取本卡的 `ParallelDiTBlock`。

    顺带负责「adaln 卸 CPU 时每步只算一次 modulation」—— `t_emb` 在一个 step
    内是同一个对象，所以第一次进来算一遍，后面 49 个 block 直接复用。
    """

    __slots__ = ("runner", "idx")

    def __init__(self, runner, idx):
        self.runner = runner
        self.idx = idx

    def __call__(self, x, t_emb, mod_segments, rope_freqs,
                 transformer_options=None, attention=None):
        return self.runner.block_forward(self.idx, x, t_emb, mod_segments,
                                         rope_freqs, transformer_options, attention)

    def __repr__(self):
        return f"<RankBlockProxy {self.idx}>"


class H3ParallelRunner:
    """一个模型 + 一组卡 = 一套并行执行环境。"""

    def __init__(self, model, device_ids, mlp_chunk=8192, offload=(), tag="h3"):
        self.model = model
        self.device_ids = list(device_ids)
        self.world = len(self.device_ids)
        if self.world < 2:
            raise ValueError("并行至少需要两张卡；单卡请直接跑原模型")
        self.devices = [torch.device(f"cuda:{i}") for i in self.device_ids]
        self.ex = HostExchange([f"cuda:{i}" for i in self.device_ids], tag=tag)
        self.mlp_chunk = mlp_chunk
        self.offload = tuple(offload)
        self.blocks = []          # blocks[rank][idx]
        self.mods = []            # mods[rank][name]
        self.cpu_once = {}        # "adaln" / "refiner" -> CpuModule
        self._mod_cache = _Memo()
        self._mod_flight = threading.Lock()
        self._installed = False
        self._orig_blocks = None
        self._orig_forward = None
        self._orig_mods = {}
        self.notes = []

    # ------------------------------------------------------------ 构建
    def _check(self):
        b0 = self.model.blocks[0]
        heads = b0.attn.heads
        ffn_half = b0.mlp.fc1.weight.shape[0] // 2
        ok, detail = check_split(heads, b0.attn.head_dim, ffn_half, self.world)
        if not ok:
            raise ValueError(f"{self.world} 卡切不开：{detail}")
        return heads, ffn_half

    def _make_copies(self, mod):
        """在 CPU 上做出 world 份拷贝，再各自搬到目标卡。

        顺序很重要：先整体 `.to("cpu")` 把显存释放掉，再从 CPU 复制。如果先
        deepcopy 再搬，峰值会是 2 份同时在 cuda:0 上（refiner 就是 2.9G）。
        """
        cpu_mod = mod.to("cpu")
        copies = [cpu_mod] + [copy.deepcopy(cpu_mod) for _ in range(self.world - 1)]
        return [m.to(dev) for m, dev in zip(copies, self.devices)]

    def _replicate_modules(self):
        """非 block 模块每卡一份；标记为可卸载的直接放 CPU 常驻。"""
        model = self.model
        self.mods = [{} for _ in range(self.world)]
        for name in NON_BLOCK:
            mod = getattr(model, name, None)
            if mod is None:
                continue
            self._orig_mods[name] = mod
            once = ONCE_MODULES.get(name)
            if once is not None and once in self.offload:
                shared = CpuModule(mod, self.devices[0], name=name)
                self.cpu_once[once] = shared
                proxy = CpuOnceProxy(shared, self.ex, self.devices, name)
                per = [proxy] * self.world
                self.notes.append(f"{name} → CPU 常驻 "
                                  f"{shared.ram_gib * 1024:.0f} MiB（带输出缓存）")
            else:
                per = self._make_copies(mod)
                proxy = RankModuleProxy(per, self.ex, name)
            for r in range(self.world):
                self.mods[r][name] = per[r]
            _set_plain(model, name, proxy)

    def _offload_adaln(self):
        """adaln 是每个 block 自己的子模块，得等并行 block 建好之后才能换。"""
        if "adaln" not in self.offload:
            return
        shared = CpuModule(self._orig_blocks[0].adaln_proj, self.devices[0],
                           name="adaln")
        self.cpu_once["adaln"] = shared
        for row in self.blocks:
            for pb in row:
                pb.adaln_proj = shared
        self.notes.append(f"adaln → CPU 常驻 {shared.ram_gib * 1024:.0f} MiB"
                          f"（每步算一次，两卡共用）")

    def build(self):
        """切分权重、装代理。**这一步之后原 block 的权重就该被释放。**"""
        self._check()
        model = self.model
        self._orig_blocks = model.blocks
        self._replicate_modules()

        self.blocks = []
        for r in range(self.world):
            dev = self.devices[r]
            row = []
            for src in self._orig_blocks:
                pb = ParallelDiTBlock(src, r, self.world, self.ex, device=dev,
                                      mlp_chunk=self.mlp_chunk)
                if r > 0:
                    # 共享的 norm / q_norm / adaln 必须各自一份，否则会把 rank0 的搬走
                    pb.norm1 = copy.deepcopy(pb.norm1).to(dev)
                    pb.norm2 = copy.deepcopy(pb.norm2).to(dev)
                    pb.q_norm = copy.deepcopy(pb.q_norm).to(dev)
                    pb.k_norm = copy.deepcopy(pb.k_norm).to(dev)
                    pb.adaln_proj = copy.deepcopy(pb.adaln_proj).to(dev)
                else:
                    pb.norm1 = pb.norm1.to(dev)
                    pb.norm2 = pb.norm2.to(dev)
                    pb.q_norm = pb.q_norm.to(dev)
                    pb.k_norm = pb.k_norm.to(dev)
                    pb.adaln_proj = pb.adaln_proj.to(dev)
                row.append(pb)
            self.blocks.append(row)
        self._offload_adaln()

        # 用代理顶替 blocks（不能直接给 nn.ModuleList 赋 list，会被 __setattr__ 拒绝）
        _set_plain(model, "blocks", [RankBlockProxy(self, i)
                                     for i in range(len(self._orig_blocks))])
        self.notes.append(
            f"{self.world} 卡张量并行：每卡 "
            f"{self.blocks[0][0].heads_local}/{self.blocks[0][0].heads} 头、"
            f"ffn {self.blocks[0][0].ffn_local}、mlp_chunk {self.mlp_chunk}")
        return self

    def free_originals(self):
        """释放原 block 的完整权重（切分完就没用了）。"""
        if self._orig_blocks is None:
            return
        for b in self._orig_blocks:
            b.to("meta")
        self._orig_blocks = None
        import gc
        gc.collect()
        for d in self.devices:
            with torch.cuda.device(d):
                torch.cuda.empty_cache()

    # ------------------------------------------------------------ 前向
    def _mod_for_step(self, t_emb, rank):
        """adaln 卸 CPU 时：一个 step 只算一次，两个 rank 共用。"""
        shared = self.cpu_once.get("adaln")
        if shared is None:
            return None                     # adaln 还在 GPU 上，block 自己算
        key = _Memo.key_of(t_emb)
        got = self._mod_cache.get(key)
        if got is None:
            with self._mod_flight:
                got = self._mod_cache.get(key)
                if got is None:
                    got = self._mod_cache.put(key, t_emb, shared.compute_cpu(t_emb))
        return tuple(t.to(self.devices[rank], non_blocking=True) for t in got)

    def block_forward(self, idx, x, t_emb, mod_segments, rope_freqs,
                      transformer_options, attention):
        rank = self.ex.rank
        mod = self._mod_for_step(t_emb, rank)
        return self.blocks[rank][idx](x, t_emb, mod_segments, rope_freqs,
                                      transformer_options, attention=attention,
                                      mod=mod)

    def _replicate_inputs(self, x, timestep, context, denoise_mask, audio_denoise_mask):
        """每 rank 一份输入。`x` 每步都变（必须搬），其余是常量、搬运可忽略。"""
        def mv(t, dev, **kw):
            if torch.is_tensor(t):
                return t.to(dev, **kw)
            if isinstance(t, (list, tuple)):
                return type(t)(mv(e, dev, **kw) for e in t)
            return t

        out = {}
        for r, dev in enumerate(self.devices):
            out[r] = dict(
                x=[t.to(dev, non_blocking=True) for t in x],
                timestep=mv(timestep, dev),
                context=mv(context, dev, non_blocking=True),
                denoise_mask=mv(denoise_mask, dev),
                audio_denoise_mask=mv(audio_denoise_mask, dev),
            )
        return out

    def install(self):
        """把 `model._forward` 换成并行版本。"""
        if self._installed:
            return self
        if not self.blocks:
            self.build()
        model = self.model
        self._orig_forward = model._forward
        runner = self

        def parallel_forward(x, timestep, context, transformer_options={},
                             minimax_payload=None, denoise_mask=None,
                             audio_denoise_mask=None, **kwargs):
            # 动态显存的预取队列不认识我们的代理 block，必须关掉
            to = dict(transformer_options)
            to.pop("prefetch_dynamic_vbars", None)
            ins = runner._replicate_inputs(x, timestep, context,
                                           denoise_mask, audio_denoise_mask)
            outs = [None] * runner.world

            def body(rank):
                a = ins[rank]
                with torch.inference_mode():
                    outs[rank] = runner._orig_forward(
                        a["x"], a["timestep"], a["context"], to,
                        minimax_payload=minimax_payload,
                        denoise_mask=a["denoise_mask"],
                        audio_denoise_mask=a["audio_denoise_mask"], **kwargs)

            runner.ex.run(body)
            # 两个 rank 结果一致（残差流复制 + 归约），取 rank0 的
            return outs[0]

        model._forward = parallel_forward
        self._installed = True
        return self

    def release(self):
        for m in self.cpu_once.values():
            m.release()
        self.cpu_once.clear()
        self.ex.release()

    def __repr__(self):
        return (f"<H3ParallelRunner {self.world} 卡 {self.device_ids} "
                f"offload={list(self.offload)}>")


def install_parallel(model, device_ids, plan=None, mlp_chunk=8192, tag="h3"):
    """一步到位：按 `Placement` 装好并行并返回 runner。"""
    offload = tuple(plan.offloaded) if plan is not None else ()
    runner = H3ParallelRunner(model, device_ids, mlp_chunk=mlp_chunk,
                              offload=offload, tag=tag)
    runner.build()
    runner.install()
    return runner
