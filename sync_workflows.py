#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同步「发布版」工作流进仓库 workflows/ 的 git 索引（符号链接布局下恢复提交能力）。

背景：
  本机 workflows/ 是指向 ComfyUI 用户工作流目录「发布版」的符号链接（单源真相，
  工作流管理界面改的就是它）。但 git 有条硬规则：不允许穿过符号链接提交里面的文件
  （git add 会报 fatal: pathspec ... is beyond a symbolic link），并且把链接下面
  的已跟踪文件全看成「已删除」。

  仓库里 workflows/ 保持真文件（GitHub 克隆即可用），本机用符号链接 + skip-worktree
  + .git/info/exclude 维持两不误。本脚本用 git plumbing（hash-object --stdin +
  update-index --cacheinfo）把「发布版」最新内容直接写进 git 索引——全程不碰
  git add，因此不受符号链接限制。

用法：
  python sync_workflows.py     # 同步内容进索引（含新增/删除），随后正常 git commit 即可
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
WORKFLOWS = os.path.join(REPO, "workflows")


def git(*args, data=None):
    """跑 git 子命令，失败时把 stderr 露出来。"""
    proc = subprocess.run(
        ["git", *args], cwd=REPO, input=data,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        sys.exit(f"[凤希AI] git {' '.join(args)} 失败：{proc.stderr.decode('utf-8', 'replace').strip()}")
    return proc.stdout.decode("utf-8", "replace").strip()


def main():
    if not os.path.isdir(WORKFLOWS):
        sys.exit("[凤希AI] workflows/ 不存在（应为指向「发布版」的符号链接或真目录）")

    is_link = os.path.islink(WORKFLOWS)
    names = sorted(n for n in os.listdir(WORKFLOWS) if n.endswith(".json") and os.path.isfile(os.path.join(WORKFLOWS, n)))
    tracked = [p[len("workflows/"):] for p in git("ls-files", "workflows").splitlines() if p.startswith("workflows/")]

    # 1) 现有文件：读内容（OS 层允许穿符号链接）→ hash-object 入库 → 写索引
    for name in names:
        with open(os.path.join(WORKFLOWS, name), "rb") as f:
            blob = git("hash-object", "-w", "--stdin", data=f.read())
        git("update-index", "--add", "--cacheinfo", f"100644,{blob},workflows/{name}")

    # 2) 发布版里已删掉的 → 从索引移除
    removed = [n for n in tracked if n not in names]
    for name in removed:
        git("update-index", "--force-remove", f"workflows/{name}")

    # 3) update-index 会清掉 skip-worktree 位，重新挂上：让 git status 平时不拿
    #    符号链接里的内容对照（否则 13 条全显示「已删除」），也防 git add -A 误删。
    #    仅符号链接布局需要；真目录布局（别人克隆）保持普通 git 行为。
    if is_link:
        for name in names:
            git("update-index", "--skip-worktree", f"workflows/{name}")

    msg = f"[凤希AI] 工作流同步：{len(names)} 条写入索引"
    if removed:
        msg += f"，移除 {len(removed)} 条"
    print(msg)
    staged = git("diff", "--cached", "--name-only")
    if staged:
        print("[凤希AI] 待提交：\n" + staged)
    else:
        print("[凤希AI] 索引与 HEAD 一致，无内容需要提交")


if __name__ == "__main__":
    main()
