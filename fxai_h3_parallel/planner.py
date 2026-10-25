"""显存预算 → 放置决策。

设计目标：**同一份程序，按模型体积自动决定拆分方式**，不靠用户选版本、也不靠模型名。
输入只有两样东西 —— 权重清单（从 checkpoint 头读，不加载权重）+ 每张卡的空闲显存，
输出一张「放哪」的计划表。将来换别的 20G / 30G 模型，只要写一个新的 classify 适配器。

三档策略，从优到劣：

  full_gpu      全部常驻显存。可切的部分等分到各卡，不可切的每卡复制一份。
  offload_once  装不下 → 把「每步只算一次」的模块（时间条件 adaln、文本 refiner）
                按体积从大到小挪到 CPU。这类模块的输入与序列长度无关，计算量极小，
                权重却可能占十几个 G，是性价比最高的卸载对象。
  stream        还装不下 → 需要每步从内存流式搬 block 权重。目前只报账，不实现。

为什么优先卸 once 而不是卸 block：block 权重每步都要完整过一遍 PCIe
（H3 两卡每卡 8 GiB / 7 GB/s ≈ +1.3 s/步），而 adaln 每步只算一次、结果只有几百 KB，
实测只慢 1.2%，还能用后台线程完全藏掉。
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

__all__ = [
    "BudgetError", "WeightGroup", "Placement",
    "read_groups", "classify_h3", "plan_placement",
]

GIB = float(1 << 30)


class BudgetError(RuntimeError):
    """显存/内存怎么摆都装不下。"""


@dataclass(frozen=True)
class WeightGroup:
    """一组同构权重（例如 50 个 block 的 fc1 算一组）。"""
    name: str
    nbytes: int                 # 单份字节数
    count: int = 1              # 有几份
    shardable: bool = False     # 能否按 world 等分
    once: bool = False          # 每步只算一次（输入与 token 数无关）→ 可卸 CPU

    @property
    def gib(self) -> float:
        return self.nbytes * self.count / GIB


@dataclass
class Placement:
    tier: str                   # full_gpu | offload_once | stream
    world: int
    per_card_gib: float         # 每卡要放的权重
    offloaded: list = field(default_factory=list)   # 挪到 CPU 的组名
    notes: list = field(default_factory=list)

    @property
    def needs_offload(self) -> bool:
        return self.tier == "offload_once"


# ------------------------------------------------------------------ checkpoint 头
def read_groups(path, classify, prefix=""):
    """只读 safetensors 头，按 classify 归类累加，返回 list[WeightGroup]。

    classify(key, dtype, shape) -> (category, shardable, once) | None
    返回 None 表示这个张量不参与预算（例如 scale / comfy_quant 之类的附属张量，
    它们的体积可以忽略，但为了准确还是建议归到 misc 里）。
    """
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))

    acc = {}          # (category, shardable, once) -> [总字节, 计数]
    blocks = set()
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        if prefix and key.startswith(prefix):
            key = key[len(prefix):]
        r = classify(key, meta["dtype"], meta["shape"])
        if r is None:
            continue
        cat, shardable, once = r
        nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
        k = (cat, bool(shardable), bool(once))
        if k not in acc:
            acc[k] = [0, 0]
        acc[k][0] += nbytes
        acc[k][1] += 1
        # 按 block 号归并成「一份」，方便 count 报出层数
        if key.startswith("blocks."):
            blocks.add(key.split(".")[1])

    # blocks.* 类的组：nbytes 是「所有层加总」，换算成单层 × count
    nblocks = len(blocks) or 1
    out = []
    for (cat, shardable, once), (total, cnt) in sorted(acc.items()):
        if cat.startswith("block.") or cat in ("adaln",):
            out.append(WeightGroup(cat, total // nblocks, nblocks, shardable, once))
        else:
            out.append(WeightGroup(cat, total, 1, shardable, once))
    return out


def classify_h3(key, dtype, shape):
    """MiniMax H3 的权重分类。

    块内**四个**大矩阵全都能切：qkv / fc1 按输出行切，out_proj / fc2 按输入列切。
    纯张量并行下 out_proj 也能切（SP 下不能 —— all_to_all 之后每卡已经有全部头），
    这一条差别就是 50 层 × 36.8 MiB = **1.80 GiB** 的每卡显存。

    adaln / token_refiner / condition_proj / time_embedder 都只吃 t_emb 或文本，
    每步算一次，是 CPU 卸载的首选。
    """
    if key.endswith("comfy_quant") or key.endswith("weight_scale"):
        return None                      # 量化附属张量，体积可忽略
    if ".adaln_proj." in key:
        return ("adaln", False, True)
    if key.startswith("blocks."):
        if (".mlp.fc1." in key or ".mlp.fc2." in key
                or ".attn.qkv_proj." in key or ".attn.out_proj." in key):
            return ("block.shard", True, False)
        return ("block.keep", False, False)          # norm / bias
    if key.startswith("token_refiner."):
        return ("refiner", False, True)              # 只在文本 token 上跑一次
    if key.startswith("final_layer."):
        return ("final", False, True)
    if key.startswith("condition_proj.") or key.startswith("time_embedder."):
        return ("cond", False, True)
    if key.startswith("video_patch_proj.") or key.startswith("audio_patch_proj."):
        return ("patch_proj", False, False)          # 每个 token 都要过，不能卸
    return ("misc", False, False)


# ------------------------------------------------------------------ 决策
def _per_card(groups, world, offloaded):
    """每卡要放的权重：可切的等分，其余全量复制，被卸掉的不算。"""
    total = 0.0
    for g in groups:
        if g.name in offloaded:
            continue
        total += g.gib / world if g.shardable else g.gib
    return total


def plan_placement(groups, world, free_per_card, activation_gib,
                   context_gib=0.3, ram_free_gib=None, safety=0.97):
    """给定模型清单和每卡空闲显存，选出最优的一档。

    free_per_card: 每张卡当前空闲（GiB）。CUDA context 已经算在里面了，不用重复扣。
    activation_gib: 单卡激活峰值（实测值）。纯张量并行下 H3 在 S=39000 实测
        `max_memory_allocated` 4.74G，减去该层 bf16 权重 0.745G → **4.0**。
        含复制的残差流（400 MiB）、qkv 输出（800 MiB）、两次 all_reduce 的显存
        暂存（各 400 MiB）。
    context_gib: 加载期间还会涨的一点余量（cuBLAS workspace 之类）
    ram_free_gib: 可用于放卸载权重的内存；None 表示不检查
    safety: 只敢用 97% 的空闲，剩下的留给分配器碎片
    """
    budget = free_per_card * safety - context_gib
    notes = []
    need_all = _per_card(groups, world, set())

    if need_all + activation_gib <= budget:
        notes.append(
            f"全部常驻：每卡权重 {need_all:.2f}G + 激活 {activation_gib:.2f}G "
            f"≤ 可用 {budget:.2f}G")
        return Placement("full_gpu", world, need_all, [], notes)

    # 按体积从大到小卸「每步只算一次」的组
    candidates = sorted((g for g in groups if g.once and g.gib > 0.05),
                        key=lambda g: -g.gib)
    if not candidates:
        raise BudgetError(
            f"每卡权重 {need_all:.2f}G + 激活 {activation_gib:.2f}G 超出可用 "
            f"{budget:.2f}G，而且没有可卸载的『每步一次』模块")

    offloaded, ram_used = set(), 0.0
    for g in candidates:
        offloaded.add(g.name)
        ram_used += g.gib
        if ram_free_gib is not None and ram_used > ram_free_gib * 0.8:
            offloaded.discard(g.name)
            ram_used -= g.gib
            continue
        need = _per_card(groups, world, offloaded)
        if need + activation_gib <= budget:
            notes.append(
                f"卸到 CPU：{', '.join(sorted(offloaded))}"
                f"（{ram_used:.2f}G 常驻内存）→ 每卡权重 {need:.2f}G "
                f"+ 激活 {activation_gib:.2f}G ≤ 可用 {budget:.2f}G")
            return Placement("offload_once", world, need, sorted(offloaded), notes)

    need = _per_card(groups, world, offloaded)
    stream_extra = sum(g.gib for g in groups
                       if g.shardable and g.name not in offloaded) / world
    raise BudgetError(
        f"卸掉 {', '.join(sorted(offloaded)) or '（无可卸项）'} 之后每卡仍需 "
        f"{need + activation_gib:.2f}G > 可用 {budget:.2f}G；"
        f"下一步只能流式搬 block 权重（每步每卡 {stream_extra:.1f}G 过 PCIe，"
        f"约 +{stream_extra / 7.0:.1f} s/步），该路径尚未实现")
