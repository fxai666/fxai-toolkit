# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权

"""凤希AI · 异常自愈守卫（显存/内存异常不再带崩服务）

为什么要这个模块
================
2026-10-30 23:25 实测：第 2 段开始 "Model Initializing" 时整条服务被打死，
`/system_stats` 返回 500、队列卡在 running 1，只能重启进程。日志链是：

    torch.AcceleratorError: CUDA error: out of memory      ← pinned_memory._steal_pin
      → sync_all_offload_streams → stream.synchronize()
    RuntimeError: aimdo memory compile error               ← 原生层把它包了一层
    RuntimeError: HostBuffer.truncate failed               ← 卸载链路二次崩

关键在第二步：`aimdo` 把 CUDA OOM 包成了一个普通 `RuntimeError`，
而 ComfyUI 的 `is_oom()` 只认 `OOM_EXCEPTION` 和 `AcceleratorError`：

    def is_oom(e):
        if isinstance(e, OOM_EXCEPTION): return True
        if isinstance(e, ACCELERATOR_ERROR) and (... "out of memory" ...): return True
        return False

于是 `execution.py` 里那段「Got an OOM, unloading all loaded models」根本不会执行 ——
本该卸载模型自愈的路径被跳过了，模型继续占着 32.4G，随后 truncate 再崩一次，
进程就废了。

四层兜底
========
L1  **认得出 OOM**：把 `is_oom()` 包一层，`aimdo memory compile error`、
    `HostBuffer.*`、`CUDA error` 也判成 OOM。这样 ComfyUI 自带的
    「打印内存摘要 + 卸载所有模型」逻辑就会正常触发 —— 错误照常报给前端，
    但进程是干净的，队列能接着跑下一个任务。

L2  **清理链路不二次崩**：`HostBuffer.truncate / unregister / register` 失败
    只记日志、不再抛。清理失败 ≠ 执行失败，不该中断整条 prompt。

L3  **硬回收 recover()**：清异步 CUDA 错误 → 卸模型 → 清显存池 → 回收 IPC
    → gc → 重置 pinned 记账 → 收缩工作集。可反复调用，幂等。

L4  **接口**：`GET /fxai/guard/status`、`POST /fxai/guard/recover`。
    服务卡住时不用重启进程，打一下 recover 就回来了：

        curl -X POST http://127.0.0.1:8188/fxai/guard/recover

环境变量
========
    FXAI_GUARD=0        关掉全部兜底（排查「是不是守卫搞的」时用）
    FXAI_GUARD_LOG=0    不打印恢复日志
"""

import gc
import os
import time
import traceback

_state = {
    "installed": False,
    "recover_count": 0,
    "last_recover_at": None,
    "last_recover_reason": "",
    "last_recover_steps": [],
    "last_error": "",
    "errors_seen": 0,
}

_ENABLED = os.environ.get("FXAI_GUARD", "1") != "0"
_VERBOSE = os.environ.get("FXAI_GUARD_LOG", "1") != "0"
# 任务跑完自动把模型和缓存放掉。默认开 —— 不放的话跑完一条 32G 模型的活儿，
# 进程会一直占着 ~69G 内存（实测 2026-10-31 00:10：跑完仍占 69.17G / 显存 6.2G）。
# 代价：下次开跑要重新 stage 一遍权重（约 1-2 分钟）。要连续调参就设 0。
_AUTO_FREE = os.environ.get("FXAI_AUTO_FREE", "1") != "0"

# 这些片段出现在异常里，就按「显存/内存不足」处理
_OOM_HINTS = (
    "aimdo memory compile error",
    "hostbuffer",
    "host buffer",
    "out of memory",
    "cuda error",
    "cuda out of memory",
    "memory allocation",
    "not enough memory",
    "pinned memory",
)


def _log(msg):
    if _VERBOSE:
        print(f"[凤希AI·守卫] {msg}", flush=True)


def looks_like_oom(ex):
    """字符串级兜底判断：`aimdo` 把 OOM 包成普通 RuntimeError，只能看文本。"""
    if ex is None:
        return False
    text = f"{type(ex).__name__}: {ex}".lower()
    return any(h in text for h in _OOM_HINTS)


# ---------------------------------------------------------------- 硬回收

def _reset_pin_accounting():
    """全部模型卸载后，pinned 记账理应归零。

    `truncate` 失败会让 `TOTAL_PINNED_MEMORY` 虚高。虚高之后每一次 pin 都会
    判成「超预算」→ 走 `_steal_pin` 分支 → 每次都撞同一个 CUDA 错误，
    等于把后续所有段都判死刑。所以回收完必须把它抹平。
    """
    import comfy.model_management as mm
    before = getattr(mm, "TOTAL_PINNED_MEMORY", 0)
    mm.TOTAL_PINNED_MEMORY = 0
    return before


def _shrink_working_set():
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.GetCurrentProcess()
        kernel32.SetProcessWorkingSetSize(handle, -1, -1)
    except Exception:
        pass


def recover(reason="手动", quiet=False, shrink=False):
    """硬回收。每一步单独 try，任何一步失败都不影响其余步骤。

    :param shrink: 是否强制收缩进程工作集（SetProcessWorkingSetSize）。
        默认 **False** —— 它会把 staged 的 32.4G 权重换出到页面文件，
        下一段重新 stage 时「模型初始化」从 1 分 15 秒涨到 28 分 51 秒
        （2026-10-31 01:25 实测）。只有手动排查/要立刻把内存还给系统时才开。
    """
    import comfy.model_management as mm
    import torch

    steps = []

    def step(name, fn):
        t0 = time.time()
        try:
            detail = fn()
            steps.append({"step": name, "ok": True,
                          "ms": int((time.time() - t0) * 1000),
                          "detail": detail})
        except Exception as e:
            steps.append({"step": name, "ok": False,
                          "ms": int((time.time() - t0) * 1000),
                          "detail": f"{type(e).__name__}: {e}"})

    step("清异步CUDA错误", mm.discard_cuda_async_error)
    step("卸载所有模型", mm.unload_all_models)
    step("清软缓存", mm.soft_empty_cache)
    step("CUDA同步", lambda: torch.cuda.synchronize() if torch.cuda.is_available() else "无CUDA")
    step("清显存池", lambda: torch.cuda.empty_cache() if torch.cuda.is_available() else "无CUDA")
    step("回收IPC句柄", lambda: torch.cuda.ipc_collect() if torch.cuda.is_available() else "无CUDA")
    step("垃圾回收", lambda: (gc.collect(), gc.collect())[1])
    step("重置pinned记账", _reset_pin_accounting)
    if shrink:
        step("收缩工作集", _shrink_working_set)
    # 最后再清一次：卸载过程本身可能又塞进新的异步错误
    step("收尾清异步错误", mm.discard_cuda_async_error)

    _state["recover_count"] += 1
    _state["last_recover_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _state["last_recover_reason"] = reason
    _state["last_recover_steps"] = steps

    bad = [s for s in steps if not s["ok"]]
    if not quiet:
        _log(f"硬回收完成（{reason}）第 {_state['recover_count']} 次"
             + (f"，{len(bad)} 步失败：{[s['step'] for s in bad]}" if bad else "，全部成功"))
    else:
        freed = [s for s in steps if s["step"] == "收缩工作集"]
        print(f"[凤希AI·守卫] 任务结束，已释放显存与内存"
              + (f"（{len(bad)} 步失败）" if bad else ""), flush=True)
    return {"ok": not bad, "reason": reason, "steps": steps}


# ---------------------------------------------------------------- 补丁

def _patch_is_oom():
    """让 ComfyUI 的 is_oom() 认得出 aimdo 包装过的 OOM。"""
    import comfy.model_management as mm
    orig = mm.is_oom
    if getattr(orig, "_fxai_guarded", False):
        return False

    def is_oom(e):
        if orig(e):
            return True
        if looks_like_oom(e):
            # 顺手把残留的异步 CUDA 错误清掉，否则下一次同步还会炸
            try:
                mm.discard_cuda_async_error()
            except Exception:
                pass
            _state["errors_seen"] += 1
            _state["last_error"] = f"{type(e).__name__}: {e}"[:500]
            return True
        return False

    is_oom._fxai_guarded = True
    mm.is_oom = is_oom
    return True


def _patch_host_buffer():
    """HostBuffer 的清理类方法失败只记日志，不抛。

    只包 truncate / unregister / register —— `extend` 有返回值（偏移量），
    包了会把「返回 False」当成偏移 0，反而更糟。
    """
    try:
        import comfy_aimdo.host_buffer as hb
    except Exception as e:
        _log(f"没找到 comfy_aimdo.host_buffer，跳过 L2：{e}")
        return False

    cls = getattr(hb, "HostBuffer", None)
    if cls is None:
        return False

    patched = False
    for name in ("truncate", "unregister", "register"):
        orig = getattr(cls, name, None)
        if orig is None or getattr(orig, "_fxai_guarded", False):
            continue

        def make(name, orig):
            def wrapper(self, *args, **kwargs):
                try:
                    return orig(self, *args, **kwargs)
                except RuntimeError as e:
                    _state["last_error"] = f"HostBuffer.{name}: {e}"[:500]
                    _log(f"HostBuffer.{name} 失败，已忽略（清理失败不影响执行）：{e}")
                    return None
            wrapper._fxai_guarded = True
            wrapper.__name__ = name
            wrapper.__doc__ = orig.__doc__
            return wrapper

        setattr(cls, name, make(name, orig))
        patched = True

    return patched


def _patch_unload():
    """unload_all_models 自己也可能抛（detach → truncate），包一层。"""
    import comfy.model_management as mm
    orig = mm.unload_all_models
    if getattr(orig, "_fxai_guarded", False):
        return False

    def unload_all_models(*a, **kw):
        try:
            return orig(*a, **kw)
        except Exception as e:
            _log(f"卸载模型时报错，已忽略：{type(e).__name__}: {e}")
            try:
                mm.soft_empty_cache()
            except Exception:
                pass
            return None

    unload_all_models._fxai_guarded = True
    mm.unload_all_models = unload_all_models
    return True


def _patch_debug_summary():
    """OOM 分支里会调 debug_memory_summary() → torch.cuda.memory_summary()。

    显存炸完那一刻这个调用本身就可能抛（上下文已脏），一抛就把整段
    「Got an OOM, unloading all loaded models」的自愈逻辑带走了 ——
    所以它必须永不失败。
    """
    import comfy.model_management as mm
    orig = getattr(mm, "debug_memory_summary", None)
    if orig is None or getattr(orig, "_fxai_guarded", False):
        return False

    def debug_memory_summary(*a, **kw):
        try:
            return orig(*a, **kw)
        except Exception as e:
            return f"（内存摘要取不到：{type(e).__name__}: {e}）"

    debug_memory_summary._fxai_guarded = True
    mm.debug_memory_summary = debug_memory_summary
    return True


def _patch_auto_free():
    """整条任务跑完（成功或失败）自动释放模型与显存/内存。

    ComfyUI 默认「跑完不卸载」，为的是下次跑得快。代价是跑完一条 32G 模型的活儿，
    进程会一直占着 69G 内存不放 —— 实测 2026-10-31 00:10。手动 `POST /free`
    能放掉（69.17G → 5.95G），这里把它做成自动的。
    """
    if not _AUTO_FREE:
        return False
    try:
        import execution
    except Exception:
        return False

    cls = getattr(execution, "PromptExecutor", None)
    if cls is None:
        return False
    orig = cls.execute
    if getattr(orig, "_fxai_guarded", False):
        return False

    def execute(self, *args, **kwargs):
        try:
            return orig(self, *args, **kwargs)
        finally:
            try:
                recover(reason="任务结束", quiet=True)
            except Exception as e:
                _log(f"任务结束释放失败：{type(e).__name__}: {e}")

    execute._fxai_guarded = True
    cls.execute = execute
    return True


def _patch_error_handler():
    """prompt 失败时先让 ComfyUI 把错误报给前端，再硬回收。

    放在 handle_execution_error 而不是 execute：前者每个失败 prompt 恰好调一次，
    而且已经拿到了 error 字典，不侵入执行循环。
    """
    try:
        import execution
    except Exception as e:
        _log(f"导入 execution 失败，跳过 L3：{e}")
        return False

    cls = getattr(execution, "PromptExecutor", None)
    if cls is None:
        return False

    orig = cls.handle_execution_error
    if getattr(orig, "_fxai_guarded", False):
        return False

    def handle_execution_error(self, prompt_id, prompt, current_outputs, executed, error, ex):
        try:
            orig(self, prompt_id, prompt, current_outputs, executed, error, ex)
        except Exception as e:
            _log(f"上报错误本身失败：{type(e).__name__}: {e}")
        finally:
            if looks_like_oom(ex):
                try:
                    recover(reason=f"prompt {str(prompt_id)[:8]} 显存/内存异常")
                except Exception as e:
                    _log(f"回收失败：{type(e).__name__}: {e}")

    handle_execution_error._fxai_guarded = True
    cls.handle_execution_error = handle_execution_error
    return True


# ---------------------------------------------------------------- 状态

def status():
    import comfy.model_management as mm
    out = dict(_state)
    out["enabled"] = _ENABLED
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            out["vram"] = {"free_gb": round(free / 2**30, 2),
                           "total_gb": round(total / 2**30, 2),
                           "device": torch.cuda.current_device()}
    except Exception as e:
        out["vram"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        out["pinned_gb"] = round(getattr(mm, "TOTAL_PINNED_MEMORY", 0) / 2**30, 2)
        out["pinned_budget_gb"] = round(getattr(mm, "MAX_PINNED_MEMORY", 0) / 2**30, 2)
    except Exception:
        pass
    return out


# ---------------------------------------------------------------- 路由

async def guard_status(request):
    from aiohttp import web
    return web.json_response(status())


async def guard_recover(request):
    from aiohttp import web
    reason = "接口调用"
    shrink = True          # 手动/守护进程调用时，把内存真的还给系统
    try:
        data = await request.json()
        reason = data.get("reason") or reason
        if "shrink" in data:
            shrink = bool(data["shrink"])
    except Exception:
        pass
    return web.json_response(recover(reason=reason, shrink=shrink))


def _register_routes():
    try:
        from server import PromptServer
        routes = getattr(getattr(PromptServer, "instance", None), "routes", None)
        if routes is None:
            return False
        routes.get("/fxai/guard/status")(guard_status)
        routes.post("/fxai/guard/recover")(guard_recover)
        return True
    except Exception as e:
        _log(f"注册路由失败：{e}")
        return False


# ---------------------------------------------------------------- 安装

def install():
    if not _ENABLED:
        print("[凤希AI·守卫] FXAI_GUARD=0，已关闭", flush=True)
        return False
    if _state["installed"]:
        return True

    done = {
        "is_oom": _patch_is_oom(),
        "host_buffer": _patch_host_buffer(),
        "unload": _patch_unload(),
        "debug_summary": _patch_debug_summary(),
        "error_handler": _patch_error_handler(),
        "auto_free": _patch_auto_free(),
        "routes": _register_routes(),
    }
    _state["installed"] = True
    _log("已安装 " + " / ".join(k for k, v in done.items() if v)
         + ("  |  跳过 " + " / ".join(k for k, v in done.items() if not v)
            if not all(done.values()) else ""))
    return True


if __name__ == "__main__":
    # 命令行直接回收：python fxai_guard.py 127.0.0.1:8188
    import json
    import sys
    import urllib.request

    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:8188"
    url = f"http://{host}/fxai/guard/recover"
    body = json.dumps({"reason": "命令行"}).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            print(json.dumps(json.loads(r.read().decode()), ensure_ascii=False, indent=1))
    except Exception as e:
        print(f"调用失败：{type(e).__name__}: {e}")
        traceback.print_exc()
