# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# 凤希AI - 自定义节点在线更新接口
# ============================================================
# 通过前端管家（设置 → 节点更新）调用，在服务器端执行 git 强制更新：
#   GET /fxai/node/status  - 检测本地版本 vs 线上版本
#   GET /fxai/node/update  - 强制更新（git fetch + reset --hard，丢弃本地改动）
#
# 服务器需已安装 git 并加入 PATH（推荐全局安装）。

import os
import subprocess
import asyncio
from aiohttp import web
from server import PromptServer

# 节点目录：本文件所在目录（即 custom_nodes/fxai-toolkit）
NODE_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_NAME = "origin"
REMOTE_BRANCH = "main"
GIT_TIMEOUT = 300  # 秒


def _git(args):
    """在节点目录执行 git 命令（同步），返回 (returncode, output)。"""
    cmd = ["git", "-C", NODE_DIR] + args
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GIT_TIMEOUT,
        )
        output = (proc.stdout or "").strip()
        error = (proc.stderr or "").strip()
        merged = output if output else error
        return proc.returncode, merged
    except FileNotFoundError:
        return -1, "未找到 git 命令，请确认服务器已安装 git 并加入 PATH"
    except subprocess.TimeoutExpired:
        return -1, "git 命令执行超时"


async def node_status(request):
    """检测本地版本与线上版本。"""
    local_commit = ""
    local_branch = ""
    remote_commit = ""
    remote_branch = ""
    remote_url = ""
    remote_reachable = False
    message = ""

    code, out = await asyncio.to_thread(_git, ["rev-parse", "--short", "HEAD"])
    if code != 0:
        return web.json_response({
            "status": "error",
            "message": f"获取本地版本失败：{out}",
            "node_dir": NODE_DIR,
        })
    local_commit = out

    code, out = await asyncio.to_thread(_git, ["rev-parse", "--abbrev-ref", "HEAD"])
    if code == 0:
        local_branch = out

    code, out = await asyncio.to_thread(_git, ["remote", "get-url", REMOTE_NAME])
    if code == 0:
        remote_url = out

    code, out = await asyncio.to_thread(_git, ["ls-remote", REMOTE_NAME, f"refs/heads/{REMOTE_BRANCH}"])
    if code == 0:
        remote_reachable = True
        parts = out.split()
        if parts:
            remote_commit = parts[0][:7]
            remote_branch = REMOTE_BRANCH
    else:
        message = f"无法连接远程仓库（ls-remote 失败）：{out}"

    return web.json_response({
        "status": "ok",
        "message": message,
        "node_dir": NODE_DIR,
        "local_commit": local_commit,
        "local_branch": local_branch,
        "remote_commit": remote_commit,
        "remote_branch": remote_branch,
        "remote_url": remote_url,
        "remote_reachable": remote_reachable,
        "has_update": bool(local_commit and remote_commit and local_commit != remote_commit),
        "git_available": True,
    })


async def node_update(request):
    """强制更新节点：git fetch + git reset --hard origin/main，丢弃所有本地改动。"""
    logs = []

    logs.append(f"[1/2] git fetch {REMOTE_NAME} ...")
    code, out = await asyncio.to_thread(_git, ["fetch", REMOTE_NAME, "--prune"])
    logs.append(out if out else ("成功" if code == 0 else "无输出"))
    if code != 0:
        logs.append("更新失败：fetch 出错，已终止（本地代码未被修改）")
        return web.json_response({"status": "error", "message": f"git fetch 失败：{out}", "logs": logs})

    logs.append(f"[2/2] git reset --hard {REMOTE_NAME}/{REMOTE_BRANCH} ...")
    code, out = await asyncio.to_thread(_git, ["reset", "--hard", f"{REMOTE_NAME}/{REMOTE_BRANCH}"])
    logs.append(out if out else ("成功" if code == 0 else "无输出"))
    if code != 0:
        logs.append("更新失败：reset --hard 出错")
        return web.json_response({"status": "error", "message": f"git reset 失败：{out}", "logs": logs})

    new_commit = ""
    code, out = await asyncio.to_thread(_git, ["rev-parse", "--short", "HEAD"])
    if code == 0:
        new_commit = out

    logs.append(f"更新完成，当前版本：{new_commit}")
    return web.json_response({"status": "ok", "message": "更新完成", "logs": logs, "new_commit": new_commit})


# 路由已统一注册在 fxai_api_utils.py
