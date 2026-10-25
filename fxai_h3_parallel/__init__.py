"""fxai 自研多卡并行（张量并行，零 NCCL，Windows 原生）。

模块分工：
    exchange  跨卡传输层（pinned host 中转 + 双流 + Barrier）
    block     并行 DiTBlock（切头 / 切 ffn，int8 逐位等价）
    devices   探测、选卡、显存预算
    planner   按模型体积决定放置策略（全 GPU / 卸 CPU / 流式）
    offload   「每步只算一次」模块的 CPU 常驻执行器
    runner    把上面这些装到已加载的 MiniMaxH3Model 上
"""
from .exchange import HostExchange, UnevenSplitError, split_sizes          # noqa: F401
from .block import (ParallelDiTBlock, ShardLinear, check_split,            # noqa: F401
                    slice_weight, split_even)
from .devices import (MultiGPUPlanError, decide, parse_device_ids,         # noqa: F401
                      plan_devices, probe_devices)
from .planner import (BudgetError, Placement, WeightGroup, classify_h3,    # noqa: F401
                      plan_placement, read_groups)
from .offload import CpuModule, move_once_modules_to_cpu                   # noqa: F401
from .runner import H3ParallelRunner, install_parallel                     # noqa: F401

__all__ = [
    "HostExchange", "UnevenSplitError", "split_sizes",
    "ParallelDiTBlock", "ShardLinear", "check_split", "slice_weight", "split_even",
    "MultiGPUPlanError", "decide", "parse_device_ids", "plan_devices", "probe_devices",
    "BudgetError", "Placement", "WeightGroup", "classify_h3", "plan_placement",
    "read_groups", "CpuModule", "move_once_modules_to_cpu",
    "H3ParallelRunner", "install_parallel",
]
