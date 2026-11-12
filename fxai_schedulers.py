# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权

# 自定义采样调度器注册处：工作流里写死的调度器名必须能在这里查到，否则报 invalid scheduler。
# beta57 原先由 RES4LYF 注册，实际只是核心 beta_scheduler 换了一组超参，那个包已移除。
# 新增调度器往 _SCHEDULERS 加一条即可，名称要与工作流里的一致。

from functools import partial

import comfy.samplers

# 名称: (sigma 生成函数, use_ms)
_SCHEDULERS = {
    "beta57": (partial(comfy.samplers.beta_scheduler, alpha=0.5, beta=0.7), True),
}

for _name, (_handler, _use_ms) in _SCHEDULERS.items():
    if _name not in comfy.samplers.SCHEDULER_HANDLERS:
        comfy.samplers.SCHEDULER_HANDLERS[_name] = comfy.samplers.SchedulerHandler(
            handler=_handler,
            use_ms=_use_ms,
        )
        comfy.samplers.SCHEDULER_NAMES.append(_name)
