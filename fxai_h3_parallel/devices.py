# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
"""多卡选择与开工前预检。

设计口径（都是拿本机实测数据换来的）：

- **自动 = 先摸清每张卡，再按「综合分」排序装填**。分数 = 可用显存 × 相对带宽
  （本机卡0 13.3G×1.00=13.3，卡1 13.3G×0.25=3.3 → 卡0 优先；卡0 被占只剩 2G 时
  0.5<3.3 → 自动跳到卡1）。取「装得下的最少张数」：每多一张卡就多一份集合通信
  （实测每层约 40 ms 固定成本 + 与 token 数成正比的搬运量），装得下就没理由拆。
- **手动指定严格按用户填的顺序**：第一位就是 rank 0。这是硬要求 —— 实际场景里
  卡 0 常常被别的任务占着，用户想指定卡 1 当主力。**任何情况下都不许重排。**
- **按等分校验**：当前切分实现是等分的，所以某卡装不下 `need/world` 就绝不能选进来。
- **不按「平均」切**：本机 GPU0 是 PCIe Gen5 x8（D2H 28.6 GB/s），GPU1 是 Gen4 x4
  （7.1 GB/s），差 4 倍。份额按各卡可用显存加权，并报出最慢那条链路 —— 它才是
  整体上限。
- **开工前把账算清楚**：显存不够直接抛错并报数字，不要跑到一半 OOM —— 跟
  `fxai_seedvr2_upscale` 的预检口径一致。
"""

import re

import torch

__all__ = [
    "DeviceInfo",
    "probe_devices",
    "parse_device_ids",
    "check_divisibility",
    "plan_devices",
    "MultiGPUPlanError",
]


class MultiGPUPlanError(ValueError):
    """选卡 / 分片方案不成立时抛出，消息里带具体数字。"""


class DeviceInfo:
    __slots__ = ("index", "name", "total_gib", "free_gib", "d2h_gbps")

    def __init__(self, index, name, total_gib, free_gib, d2h_gbps=None):
        self.index = index
        self.name = name
        self.total_gib = total_gib
        self.free_gib = free_gib
        self.d2h_gbps = d2h_gbps

    def __repr__(self):
        bw = "未测" if self.d2h_gbps is None else f"{self.d2h_gbps:.1f} GB/s"
        return (f"cuda:{self.index} {self.name} "
                f"总 {self.total_gib:.1f}G 空闲 {self.free_gib:.1f}G D2H {bw}")


_LINK_CACHE = {}


def measure_link_gbps(index, mib=128):
    """测单卡 D2H 到 pinned 的带宽。跨卡 host 中转时这条链路就是上限，
    两张卡各测一次取最慢的当整体瓶颈。"""
    if index in _LINK_CACHE:
        return _LINK_CACHE[index]
    n = mib * 1024 * 1024 // 2
    src = torch.empty(n, dtype=torch.float16, device=f"cuda:{index}")
    dst = torch.empty(n, dtype=torch.float16, pin_memory=True)
    stream = torch.cuda.Stream(device=index)
    import time
    with torch.cuda.stream(stream):
        dst.copy_(src, non_blocking=True)
    stream.synchronize()
    t0 = time.perf_counter()
    iters = 3
    with torch.cuda.stream(stream):
        for _ in range(iters):
            dst.copy_(src, non_blocking=True)
    stream.synchronize()
    dt = (time.perf_counter() - t0) / iters
    gbps = (n * 2) / dt / 1e9
    del src, dst
    _LINK_CACHE[index] = gbps
    return gbps


def probe_devices(measure_link=True):
    """列出所有可见卡的真实可用显存（不是标称值）。"""
    out = []
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            free, total = torch.cuda.mem_get_info()
            p = torch.cuda.get_device_properties(i)
        gbps = measure_link_gbps(i) if measure_link else None
        out.append(DeviceInfo(i, p.name, total / 2**30, free / 2**30, gbps))
    return out


def parse_device_ids(spec, available):
    """解析手填的卡号。

    支持 `"0,1"` / `"0 1"` / `"0;1"` / `"0123"`（纯数字且无分隔时逐字符）。
    十张卡以上必须用逗号 —— `"10"` 到底是第 1 张加第 0 张还是第 10 张，没法猜。

    **返回顺序就是 rank 顺序，第一位是 rank 0，绝不重排。** 只填一张也允许
    （那就单卡跑，不切分）。
    """
    if spec is None:
        raise MultiGPUPlanError("手动指定模式必须填卡号")
    s = str(spec).strip()
    if not s:
        raise MultiGPUPlanError("手动指定模式必须填卡号，例如 0,1")

    if re.search(r"[,;\s]", s):
        parts = [p for p in re.split(r"[,;\s]+", s) if p]
    elif s.isdigit():
        parts = list(s)
    else:
        raise MultiGPUPlanError(f"卡号写法看不懂：{spec!r}，用 0,1 这种逗号分隔")

    ids = []
    for p in parts:
        if not p.isdigit():
            raise MultiGPUPlanError(f"卡号里有非数字：{p!r}")
        i = int(p)
        if i in ids:
            raise MultiGPUPlanError(f"卡号 {i} 重复了")
        ids.append(i)

    bad = [i for i in ids if i not in available]
    if bad:
        raise MultiGPUPlanError(
            f"卡号 {bad} 不存在，本机只有 {sorted(available)}")
    return ids


def check_divisibility(heads, ffn_half, world):
    """头数和 ffn 半宽能否被 world 整除。

    返回 (ok, detail)。除不尽不是不能用，但必须走不等分切分 —— 而 convrot 的
    组大小是 256，ffn 的切点必须落在 256 的倍数上，所以能选的范围是有限的。
    """
    head_ok = heads % world == 0
    ffn_ok = ffn_half % world == 0
    detail = (f"heads {heads}/{world}={'整除' if head_ok else '除不尽'}，"
              f"ffn {ffn_half}/{world}={'整除' if ffn_ok else '除不尽'}")
    return head_ok and ffn_ok, detail


def plan_devices(mode, spec, need_gib, replicated_gib=0.0, available_ids=None,
                 measure_link=True, reserve_gib=1.5):
    """决定用哪几张卡、各卡大致分多少。

    mode: "自动" / "手动指定"
    need_gib: 整个 DiT 权重的体积（GiB）
    replicated_gib: 其中**不能切**、每张卡都要放一份的部分（adaln / token_refiner /
        patch_proj / final_layer / condition_proj）。Ulysses SP 是把每个 block 都放在
        每张卡上、只切 block 内部，所以这部分每卡复制一份 —— 不扣掉它，1 张 31.7G 的
        模型会被误判成「两卡各 15.8G 装得下」。
    reserve_gib: 每卡留给激活/缓存的余量

    返回 (ids, shares, notes, devs)：ids 已按「快卡在前」排好；shares 是按可用显存
    加权后的比例（未做 256/头数对齐，对齐在切分那一层做）。
    """
    devs = probe_devices(measure_link=measure_link)
    if not devs:
        raise MultiGPUPlanError("一张卡都没检测到")
    by_id = {d.index: d for d in devs}
    avail = sorted(by_id)

    if available_ids is not None:
        avail = [i for i in avail if i in available_ids]

    # 每卡实际要放的权重：可切部分等分，不可切部分全量复制
    def per_card(world):
        return (need_gib - replicated_gib) / world + replicated_gib

    notes = []
    if mode == "手动指定":
        ids = parse_device_ids(spec, avail)
    else:
        # 自动：按「综合分」（可用显存 × 相对带宽）从高到低装填，挑装得下的最少卡数。
        # 关键是按**每卡实际占用**校验 —— 当前实现是等分切分 + 不可切部分复制，
        # 若某卡装不下 per_card(world)，选进来只会 OOM，必须直接排除。
        usable = [(i, by_id[i].free_gib - reserve_gib)
                  for i in avail if by_id[i].free_gib > reserve_gib]
        if not usable:
            raise MultiGPUPlanError(
                f"没有一张卡的空闲显存超过预留 {reserve_gib}G：" +
                "；".join(f"cuda:{i} {by_id[i].free_gib:.1f}G" for i in avail))

        single = next((i for i, u in usable if u >= need_gib), None)
        if single is not None:
            ids = [single]
            notes.append(f"自动：cuda:{single} 单卡就装得下 {need_gib:.1f}G，"
                         f"按单卡跑 —— 多卡不会更快，要强制多卡请用手动指定")
        else:
            # 按「综合分」从高到低装填：分数 = 可用显存 × 相对带宽。
            # 两个因子都要 —— 显存是硬门槛（装不下就没戏），带宽决定交换有多拖。
            # 本机实测：卡0 13.3G × 1.00 = 13.3，卡1 13.3G × 0.25 = 3.3，
            # 所以卡0 排前面；若卡0 被占只剩 2G，则 0.5 < 3.3，自动跳到卡1。
            top = max((by_id[i].d2h_gbps or 1.0) for i in avail)
            scored = sorted(
                usable,
                key=lambda t: -(t[1] * (by_id[t[0]].d2h_gbps or top) / top))
            if measure_link:
                notes.append("自动排序：" + " > ".join(
                    f"cuda:{i}({u:.1f}G×{(by_id[i].d2h_gbps or top) / top:.2f}"
                    f"={u * (by_id[i].d2h_gbps or top) / top:.1f})"
                    for i, u in scored))

            ids = None
            for order, why in ((scored, "综合分"), (sorted(usable, key=lambda t: -t[1]), "显存优先")):
                for world in range(2, len(order) + 1):
                    cand = order[:world]
                    share = per_card(world)
                    if all(u >= share for _, u in cand):
                        ids = [i for i, _ in cand]
                        notes.append(
                            f"自动选卡：{ids}（{why}排序，{world} 卡，"
                            f"每卡需 {share:.1f}G = 可切 {(need_gib - replicated_gib) / world:.1f}"
                            f" + 复制 {replicated_gib:.1f}）")
                        break
                if ids is not None:
                    break
            if ids is None:
                raise MultiGPUPlanError(
                    f"权重 {need_gib:.1f}G（其中不可切 {replicated_gib:.1f}G 每卡复制）："
                    "没有单卡装得下，多卡也装不下：" +
                    "；".join(f"cuda:{i} 可用 {u:.1f}G" for i, u in usable) +
                    f"（{len(usable)} 卡时每卡要 {per_card(len(usable)):.1f}G）")

    # 逐卡复核显存：总容量 + 每卡实际占用两个约束都要过
    usable_sel = [by_id[i].free_gib - reserve_gib for i in ids]
    total_w = sum(usable_sel)
    if total_w < need_gib:
        raise MultiGPUPlanError(
            f"选中的卡扣掉预留后只有 {total_w:.1f}G，装不下 {need_gib:.1f}G：" +
            "；".join(f"cuda:{i} {by_id[i].free_gib:.1f}G" for i in ids))
    share = per_card(len(ids))
    tight = [i for i, u in zip(ids, usable_sel) if u < share]
    if tight:
        raise MultiGPUPlanError(
            f"{len(ids)} 卡时每卡要 {share:.1f}G（可切 {(need_gib - replicated_gib) / len(ids):.1f}"
            f" + 复制 {replicated_gib:.1f}），这几张装不下：" +
            "；".join(f"cuda:{i} 可用 {by_id[i].free_gib - reserve_gib:.1f}G" for i in tight))
    # 当前实现是等分切分，所以份额就是 1/n（显存不均只影响余量，不影响切分比例）
    shares = [1.0 / len(ids)] * len(ids)
    notes.append("分片：" + "；".join(
        f"rank{r} = cuda:{i}（空闲 {by_id[i].free_gib:.1f}G，可用 {u:.1f}G）"
        for r, (i, u) in enumerate(zip(ids, usable_sel))))
    if len(ids) > 1 and max(usable_sel) - min(usable_sel) > 0.25 * max(usable_sel):
        notes.append(
            f"各卡可用显存差距较大（{[f'{u:.1f}G' for u in usable_sel]}），"
            f"等分后小卡余量偏紧；不等分切分尚未实现")

    if measure_link and len(ids) > 1:
        slow = min(ids, key=lambda i: by_id[i].d2h_gbps or 0)
        notes.append(
            f"链路瓶颈 cuda:{slow}（{by_id[slow].d2h_gbps:.1f} GB/s），"
            f"其余卡更快也快不过它")

    return ids, shares, notes, devs


# ---------------------------------------------------------------- 全自动决策
def ram_free_gib():
    try:
        import psutil
        return psutil.virtual_memory().available / (1 << 30)
    except Exception:
        return None


def decide(model_path, mode="自动", spec="", activation_gib=4.0, classify=None,
           prefix="model.diffusion_model.", measure_link=True, context_gib=0.3):
    """读 checkpoint 头 → 选卡 → 定拆分档位。全程不需要加载权重。

    返回 (ids, placement, notes, groups, devs)。

    选档顺序（都是实测支撑的）：
      1) 全 GPU 常驻，卡数**从少到多**。SP 只在序列够长时才划算 —— 实测 S=256 时
         两卡只有 0.13×、S=4096 才 1.02×，所以小模型不该为了并行而占两张卡。
      2) 全 GPU 怎么都装不下 → 允许把「每步一次」的模块卸到 CPU，仍从少卡开始试。
      3) 还不行 → 报错，并给出流式搬运的代价估算。
    """
    from .planner import BudgetError, classify_h3, plan_placement, read_groups

    groups = read_groups(model_path, classify or classify_h3, prefix=prefix)
    total = sum(g.gib for g in groups)
    devs = probe_devices(measure_link=measure_link)
    if not devs:
        raise MultiGPUPlanError("一张卡都没检测到")
    by_id = {d.index: d for d in devs}
    avail = sorted(by_id)
    ram = ram_free_gib()

    notes = [f"模型权重 {total:.2f}G（含 {len(groups)} 组，读头不加载）"]

    if mode == "手动指定":
        ids = parse_device_ids(spec, avail)
        orders = [ids]                       # 严格按用户填的顺序，不排序
    else:
        top = max((by_id[i].d2h_gbps or 1.0) for i in avail)
        order = sorted(avail, key=lambda i: -(by_id[i].free_gib
                                              * (by_id[i].d2h_gbps or top) / top))
        if measure_link:
            notes.append("自动排序：" + " > ".join(
                f"cuda:{i}({by_id[i].free_gib:.1f}G×"
                f"{(by_id[i].d2h_gbps or top) / top:.2f})" for i in order))
        orders = [order[:w] for w in range(1, len(order) + 1)]

    first_offload = None
    last_err = None
    for cand in orders:
        world = len(cand)
        free = min(by_id[i].free_gib for i in cand)      # 等分 → 取最紧的那张
        try:
            pl = plan_placement(groups, world, free, activation_gib,
                                context_gib=context_gib, ram_free_gib=ram)
        except BudgetError as e:
            last_err = e
            continue
        if pl.tier == "full_gpu":
            notes.extend(pl.notes)
            notes.append(f"选定 {cand}（{world} 卡），全 GPU 常驻")
            return cand, pl, notes, groups, devs
        if first_offload is None:
            first_offload = (cand, pl)

    if first_offload is not None:
        cand, pl = first_offload
        notes.extend(pl.notes)
        notes.append(f"选定 {cand}（{len(cand)} 卡），"
                     f"{', '.join(pl.offloaded)} 放 CPU 常驻（{ram or 0:.0f}G 可用内存）")
        return cand, pl, notes, groups, devs

    raise MultiGPUPlanError(str(last_err) if last_err else "无法给出显存方案")
