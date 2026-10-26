"""fxai 自研多卡并行（零 NCCL，Windows 原生）。

骨架与模型无关，换模型只需另写一个 runner：

    通用层
      exchange  跨卡传输层（pinned host 中转 + 双流 + Barrier）
      block     ParallelDiTBlock / ShardLinear（切头、切 ffn，int8 逐位等价）
      devices   探测、选卡、显存预算
      planner   按权重体积决定放置（全 GPU / 卸 CPU / 流式）
      offload   「每步只算一次」模块的 CPU 常驻执行器

    MiniMax H3 绑定
      planner.classify_h3   H3 的权重分类器
      runner                H3ParallelRunner / install_parallel

接新模型 = 写一个 classify_xxx + 一个 Runner，上面五层原样复用。
"""
from .exchange import HostExchange, UnevenSplitError, split_sizes          # noqa: F401
from .block import (ParallelDiTBlock, ShardLinear, check_split,            # noqa: F401
                    slice_weight, split_even)
from .devices import (MultiGPUPlanError, decide, decide_from_model,        # noqa: F401
                      parse_device_ids, plan_devices, probe_devices)
from .planner import (BudgetError, Placement, WeightGroup, classify_h3,    # noqa: F401
                      groups_from_model, plan_placement, read_groups)
from .offload import CpuModule, move_once_modules_to_cpu                   # noqa: F401
from .runner import H3ParallelRunner, install_parallel                     # noqa: F401

__all__ = [
    "HostExchange", "UnevenSplitError", "split_sizes",
    "ParallelDiTBlock", "ShardLinear", "check_split", "slice_weight", "split_even",
    "MultiGPUPlanError", "decide", "decide_from_model", "parse_device_ids",
    "plan_devices", "probe_devices",
    "BudgetError", "Placement", "WeightGroup", "classify_h3", "groups_from_model",
    "plan_placement", "read_groups", "CpuModule", "move_once_modules_to_cpu",
    "H3ParallelRunner", "install_parallel",
]
