# -*- coding: utf-8 -*-
r"""
服务控制（给启动器网页底部按钮用）：
放在自定义节点包里（custom_nodes/fxai-toolkit/），跟节点一起同步更新，
不用在 ComfyUI 根目录另放一份。

    ctl.cmd kill     按 fxai.json 里的端口结束服务进程（只停不重拉）
    ctl.cmd start    按 fxai.json 里的 api / app 启动（前台运行，日志进当前终端）
    ctl.cmd where    打印定位到的 ComfyUI 根目录、配置文件和控制脚本路径

网页「重启服务」的流程就是：先用 kill 把旧进程停掉，再往同一个命令行窗口发 start，
所以这里只管停和起，不干别的。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def log(msg=""):
    print(msg, flush=True)


def read_json(p, default=None):
    try:
        with open(p, encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return default


def find_root():
    """向上找带 fxai.json 的那一层，就是 ComfyUI 根目录。

    脚本可能被放在两个地方：
      <root>/ctl.py                              自己就是根
      <root>/custom_nodes/fxai-toolkit/ctl.py    向上两级才是根
    """
    for d in [HERE] + list(HERE.parents)[:4]:
        if (d / "fxai.json").is_file():
            return d
    return HERE


ROOT = find_root()


def cfg():
    return read_json(ROOT / "fxai.json", {}) or {}


def port():
    try:
        return int(cfg().get("port") or 0)
    except Exception:
        return 0


def listen_pids(p):
    """占用该端口且处于 LISTENING 的进程号。"""
    out = set()
    try:
        r = subprocess.run(["netstat", "-ano", "-p", "TCP"],
                           capture_output=True, timeout=20)
        text = r.stdout.decode("mbcs", "replace")
    except Exception:
        return out
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[3].upper() == "LISTENING" and parts[1].endswith(":%d" % p):
            try:
                out.add(int(parts[4]))
            except ValueError:
                pass
    return out


def do_kill():
    p = port()
    if not p:
        log("%s 里没有 port，无法定位进程" % (ROOT / "fxai.json"))
        return 2
    pids = listen_pids(p)
    if not pids:
        log("端口 %d 没被占用，服务本来就没在跑" % p)
        return 0
    for pid in sorted(pids):
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True, timeout=30)
    log("已结束进程：" + ", ".join(str(x) for x in sorted(pids)))
    return 0


def do_start():
    entry = (cfg().get("api") or cfg().get("app") or "").strip()
    if not entry:
        log("%s 里没有 api / app 启动项" % (ROOT / "fxai.json"))
        return 2
    parts = entry.split()
    # 入口脚本按根目录解析成绝对路径：工作目录切到根目录，main.py 里的相对路径才不乱
    head = Path(parts[0])
    if not head.is_absolute():
        head = ROOT / head
    py = sys.executable
    argv = [py, str(head)] + parts[1:]
    log("[ctl] cwd=" + str(ROOT))
    log("[ctl] " + " ".join(argv))
    os.chdir(ROOT)
    os.execv(py, argv)          # 前台替换进程，日志继续打在当前终端里
    return 0


def do_where():
    log("root  : %s" % ROOT)
    log("config: %s" % (ROOT / "fxai.json"))
    log("script: %s" % Path(__file__).resolve())
    log("port  : %s" % (port() or "(未配置)"))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if len(sys.argv) < 2:
        log(__doc__)
        sys.exit(2)
    cmd = sys.argv[1].lower()
    if cmd == "where":
        sys.exit(do_where())
    sys.exit(do_kill() if cmd in ("kill", "stop") else do_start())
