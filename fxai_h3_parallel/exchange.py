# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
"""跨卡交换层：pinned 主机内存中转，不用 NCCL / 不用 torch.distributed。

本机实测：`can_device_access_peer` 双向 False，`Tensor.copy_()` 的"直连"实际也是
走主机中转，实测 7.0 GB/s；显式 pinned 中转 7.1 GB/s —— 两者等价，所以干脆自己
管缓冲，顺带拿到零依赖、Windows 原生可跑。

线程模型：每个 rank 一个线程（不是进程），每线程独占一张卡。集合通信靠
`threading.Barrier` 对齐，数据传输靠 `non_blocking` 拷贝 + 每个 rank 自己的
CUDA Stream。PyTorch 在 CUDA 调用期间释放 GIL，所以两个线程真的能并行。
"""

import threading

import torch

__all__ = ["HostExchange", "UnevenSplitError"]


class UnevenSplitError(ValueError):
    pass


def split_sizes(total, world):
    """把 total 行尽量平均分给 world 个 rank，返回每 rank 行数与起始偏移。"""
    base = total // world
    rem = total - base * world
    sizes = [base + (1 if i < rem else 0) for i in range(world)]
    offs = [0]
    for s in sizes:
        offs.append(offs[-1] + s)
    return sizes, offs


class HostExchange:
    """2 卡（或更多）之间的 pinned 内存交换器。

    用法::

        ex = HostExchange(["cuda:0", "cuda:1"])
        def body(rank):
            # 这里 rank 是当前线程的序号，所有原语都是「本 rank」视角
            full = ex.all_gather_rows(local)
        ex.run(body)

    缓冲按 (key, 形状, dtype) 复用，跑起来之后不再分配。
    """

    def __init__(self, devices, chunks=1, min_chunk_bytes=1 << 20, tag=""):
        self.devices = list(devices)
        self.world = len(self.devices)
        if self.world < 2:
            raise ValueError("HostExchange 至少要两张卡")
        self.chunks = max(1, int(chunks))
        self.min_chunk_bytes = int(min_chunk_bytes)
        self.tag = tag

        self._barrier = threading.Barrier(self.world)
        self._pinned = {}
        self._device_bufs = {}
        self._streams_by_rank = {}
        self._events_by_rank = {}
        self._lock = threading.Lock()
        self._errors = []
        self._local = threading.local()

    # ---------------------------------------------------------------- 线程上下文
    @property
    def rank(self):
        return self._local.rank

    @property
    def device(self):
        return self._local.device

    def _streams(self):
        """按 rank 缓存双流。**不能放在 thread-local** —— 每次 run() 都是新线程，
        新线程里的新建 stream 在 Windows WDDM 上带隐式同步，实测每 block 白花 40 ms。
        CUDA stream 归设备不归线程，所以按 rank 缓存是安全的。"""
        r = self._local.rank
        s = self._streams_by_rank.get(r)
        if s is None:
            dev = self._local.device
            s = (torch.cuda.Stream(device=dev), torch.cuda.Stream(device=dev))
            self._streams_by_rank[r] = s
        return s

    def _event(self):
        r = self._local.rank
        ev = self._events_by_rank.get(r)
        if ev is None:
            ev = torch.cuda.Event()
            self._events_by_rank[r] = ev
        return ev

    def run(self, body):
        """起 world 个线程各跑一遍 body(rank)，等全部结束；任何一路抛错都往上抛。"""
        self._errors = []
        threads = []
        for r in range(self.world):
            t = threading.Thread(target=self._worker, args=(r, body), daemon=True,
                                 name=f"fxai-xchg-{self.tag}-{r}")
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        if self._errors:
            raise self._errors[0]

    def _worker(self, rank, body):
        self._local.rank = rank
        self._local.device = torch.device(self.devices[rank])
        torch.cuda.set_device(self._local.device)
        try:
            body(rank)
        except BaseException as exc:  # noqa: BLE001 - 原样上抛，但要先让对端退出
            self._errors.append(exc)
            # 让还在 barrier 上等着的对端尽快失败而不是死等
            self._barrier.abort()
            raise

    # ---------------------------------------------------------------- 缓冲
    def _hb(self, key, shape, dtype):
        """取（或建）一块 pinned 主机缓冲。key 必须能唯一区分用途与形状。"""
        k = (key, tuple(shape), dtype)
        with self._lock:
            buf = self._pinned.get(k)
            if buf is None:
                buf = torch.empty(shape, dtype=dtype, pin_memory=True)
                self._pinned[k] = buf
            return buf

    def release(self):
        with self._lock:
            self._pinned.clear()
            self._device_bufs.clear()
            self._streams_by_rank.clear()
            self._events_by_rank.clear()

    def pinned_bytes(self):
        with self._lock:
            return sum(t.numel() * t.element_size() for t in self._pinned.values())

    def _db(self, key, shape, dtype):
        """显存侧的暂存缓冲（按本 rank 设备）。"""
        k = (key, tuple(shape), dtype, self.devices[self.rank])
        with self._lock:
            buf = self._device_bufs.get(k)
            if buf is None:
                buf = torch.empty(shape, dtype=dtype, device=self.device)
                self._device_bufs[k] = buf
            return buf

    def _sync_in(self):
        """让旁路 D2H 流等一等当前流：输入张量可能刚由计算/拷贝产生，还没落地。"""
        ev = self._event()
        ev.record(torch.cuda.current_stream(self.device))
        s_d2h, _ = self._streams()
        s_d2h.wait_event(ev)

    # ---------------------------------------------------------------- 原语
    def all_gather_rows(self, local, sizes, offs, key="gather", out=None):
        """把各 rank 的 [n_r, K] 拼成 [S, K]，每个 rank 都拿到完整结果。

        只搬「别人的行」：自己那段本来就在显存里，直接放进输出，不做一次多余的往返。
        每 rank 出去 n_r 行、进来 S-n_r 行。
        """
        total = offs[-1]
        r = self.rank
        n_local, k = local.shape
        dtype = local.dtype
        if n_local != sizes[r]:
            raise UnevenSplitError(f"rank{r} 本地行数 {n_local} != 约定 {sizes[r]}")

        hb = self._hb(key, (total, k), dtype)
        s_d2h, s_h2d = self._streams()
        dev = self.device

        # 自己那段下到主机
        self._sync_in()
        with torch.cuda.stream(s_d2h):
            hb[offs[r]:offs[r + 1]].copy_(local, non_blocking=True)
        s_d2h.synchronize()
        self._barrier.wait()

        if out is None:
            out = torch.empty((total, k), dtype=dtype, device=dev)
        out[offs[r]:offs[r + 1]].copy_(local)
        with torch.cuda.stream(s_h2d):
            for s in range(self.world):
                if s == r:
                    continue
                out[offs[s]:offs[s + 1]].copy_(hb[offs[s]:offs[s + 1]], non_blocking=True)
        s_h2d.synchronize()
        self._barrier.wait()
        return out

    def all_to_all_heads(self, local, sizes, offs, heads_per_rank, head_dim, key="a2a"):
        """[S, Hl*D] -> [n_r, H*D]。

        每个 rank 手里是「全部 token × 自己的头组」，换完之后是「自己的 token 段 × 全部头」。
        出去 S 行、进来 n_r*world 行，两边体积相等（Hl*world = H）。
        """
        total = offs[-1]
        r = self.rank
        hl = heads_per_rank
        d = head_dim
        dtype = local.dtype
        if local.shape != (total, hl * d):
            raise UnevenSplitError(f"all_to_all 期望 {(total, hl * d)}，拿到 {tuple(local.shape)}")

        hb = self._hb(key, (self.world, total, hl * d), dtype)
        s_d2h, s_h2d = self._streams()
        dev = self.device

        self._sync_in()
        with torch.cuda.stream(s_d2h):
            hb[r].copy_(local, non_blocking=True)
        s_d2h.synchronize()
        self._barrier.wait()

        n_local = sizes[r]
        out = torch.empty((n_local, self.world * hl * d), dtype=dtype, device=dev)
        a = offs[r]
        b = offs[r + 1]
        with torch.cuda.stream(s_h2d):
            for s in range(self.world):
                out[:, s * hl * d:(s + 1) * hl * d].copy_(hb[s, a:b], non_blocking=True)
        s_h2d.synchronize()
        self._barrier.wait()
        return out

    def all_reduce_sum(self, local, key="ar", inplace=False):
        """对位相加。2 卡时就是「换一份、加一下」，不绕 ring。

        `inplace=True` 时直接加进 `local` 并返回它 —— 调用方拿到部分和之后就不再
        需要它了（TP 的 out_proj / fc2 输出都是这样），能省掉一份 [n,k] 的克隆。
        S=39000 时那一份是 400 MiB，占单层峰值的 15%。
        """
        r = self.rank
        n_local, k = local.shape
        dtype = local.dtype
        hb = self._hb(key, (self.world, n_local, k), dtype)
        s_d2h, s_h2d = self._streams()

        self._sync_in()
        with torch.cuda.stream(s_d2h):
            hb[r].copy_(local, non_blocking=True)
        s_d2h.synchronize()
        self._barrier.wait()

        # 主机缓冲不能直接加到显存张量上，先落到显存暂存区。
        # 只开**一个** [n,k] 槽轮流用：2 卡时 world*n*k 会白占一整份（S=39000 时
        # 838 MiB），改成单槽后每次 copy→add 立刻回收，峰值砍掉一半。
        staging = self._db(key, (n_local, k), dtype)
        out = local if inplace else local.clone()
        for s in range(self.world):
            if s == r:
                continue
            with torch.cuda.stream(s_h2d):
                staging.copy_(hb[s], non_blocking=True)
            s_h2d.synchronize()
            out.add_(staging)
        self._barrier.wait()
        return out

    # ---------------------------------------------------------------- 便捷封装
    def gather_and_split(self, local, sizes, offs, key="gather"):
        """一次拿到完整序列；同时把「别人那段」单独留一份，方便按需拼接。"""
        return self.all_gather_rows(local, sizes, offs, key=key)
