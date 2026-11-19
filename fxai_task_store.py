import datetime
import os
import sqlite3
import threading
import folder_paths
from aiohttp import web
from server import PromptServer

DB_DIR = os.path.join(folder_paths.base_path, "fxai")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "fxai.db")
_local = threading.local()

def _get_conn():
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH)
        _local.conn.execute("PRAGMA journal_mode=DELETE")
        _local.conn.execute("""
            CREATE TABLE IF NOT EXISTS completed_tasks (
                prompt_id TEXT PRIMARY KEY,
                workflow_id TEXT DEFAULT '',
                url TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now','localtime'))
            )
        """)
        _local.conn.commit()
    return _local.conn

def _get_current_prompt_id():
    """取「正在执行这个节点的」那条 prompt 的 prompt_id。

    ① 首选 ComfyUI 官方的执行上下文：execution.py 在调用节点 FUNCTION 时用
       CurrentNodeContext(prompt_id, ...) 设了 contextvar，这里读到的就是本节点的
       真实归属，**多条任务并发执行也不会串号**。
    ② 老版本没有 comfy_execution.utils 时退回扫队列；注意这版 ComfyUI 的
       currently_running 是字典（可以同时跑多条），只有「确实只有一条在跑」时
       才认它，多条时宁可不广播也不张冠李戴（把 A 任务的图记到 B 任务名下）。
    """
    try:
        from comfy_execution.utils import get_executing_context
        ctx = get_executing_context()
        if ctx is not None and getattr(ctx, "prompt_id", None):
            return ctx.prompt_id
    except Exception:
        pass
    try:
        running, _ = PromptServer.instance.prompt_queue.get_current_queue()
        running = [r for r in (running or []) if isinstance(r, (list, tuple)) and len(r) > 1]
        if len(running) == 1:
            return running[0][1]
        if len(running) > 1:
            print(f"[凤希AI] 广播跳过：同时有 {len(running)} 条任务在执行，无法确定产物归属")
    except Exception:
        pass
    return ""

def save_result(category, subdir, files, prompt_id=None):
    """统一保存任务结果：持久化 + WS 广播。

    Args:
        category: 类别 image/audio/video
        subdir:   子目录名
        files:    文件名列表
        prompt_id: ComfyUI prompt_id（为空时自动从队列获取）
    """
    if not prompt_id:
        prompt_id = _get_current_prompt_id()
    # 兜底：只认真实产物文件名。PreviewImage 之类写进 temp 的预览名（ComfyUI_temp_*）
    # 不是成品，进了广播会让下游按它拼下载令牌，必 404。
    files = [f for f in (files or [])
             if f and not os.path.basename(str(f)).lower().startswith("comfyui_temp")]
    if not files:
        return
    if not prompt_id:
        # 以前这里静默 return：结果既不广播也不写库，事后用 /fxai/tasks/result
        # 拿 prompt_id 去查只会是空的——外面看起来像「图生成了但没结果」。
        print("[凤希AI] 广播跳过：未取到 prompt_id，本次产物（%s）未写库，稍后按 id 查也为空"
              % ", ".join(map(str, files)))
        return
    dir_path = category + ("/" + subdir.replace("\\", "/") if subdir else "")
    url = dir_path + "|" + ",".join(files)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        PromptServer.instance.send_sync("fxai:task_saved", {
            "prompt_id": prompt_id,
            "url": url,
            "time": now
        })
    except:
        pass
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO completed_tasks (prompt_id, url) VALUES (?,?)",
        (prompt_id, url)
    )
    conn.commit()

def broadcast(event, data=None, prompt_id=None):
    """过程广播：不持久化，仅通过 WS 把「事件名 + 动态数据」实时发到前端。

    与 save_result 的区别：save_result 是任务结果（落库+广播一次）；broadcast 是
    执行过程中的事件流，前端按 data["event"] 自行拆分处理。

    Args:
        event: 调用方自定义的事件名（不同场景用不同名），前端据此分发
        data:  动态数据（dict，JSON 序列化随事件下发），由各调用方自行约定字段
        prompt_id: 任务ID（为空时自动从队列获取当前运行的 prompt_id）
    """
    if not prompt_id:
        prompt_id = _get_current_prompt_id()
    try:
        PromptServer.instance.send_sync("fxai:progress", {
            "prompt_id": prompt_id,
            "event": event,
            "data": data if data is not None else {}
        })
    except:
        pass

def get_tasks(prompt_ids):
    """批量查询任务结果。prompt_ids: 逗号分隔的字符串"""
    if not prompt_ids:
        return {}
    ids = [p.strip() for p in prompt_ids.split(",") if p.strip()]
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    conn = _get_conn()
    cur = conn.execute(
        f"SELECT * FROM completed_tasks WHERE prompt_id IN ({placeholders})", ids
    )
    rows = cur.fetchall()
    return {
        row[0]: {
            "prompt_id": row[0],
            "workflow_id": row[1],
            "url": row[2] or '',
            "created_at": row[3]
        }
        for row in rows
    }

def list_tasks(limit=50, offset=0, category="", keyword="", since="", until=""):
    """按条件分页列出任务库（持久化历史，ComfyUI 重启不丢）。

    ComfyUI 官方 /history 是 PromptServer 内存里的字典，进程一重启就清空；
    本表是落盘记录，供控制台「任务记录」页翻页 / 各生成服务查自己那类记录。

    Args:
        category: image / audio / video，空 = 全部（按 url 令牌的类别前缀过滤）
        keyword:  关键字，匹配「子目录/文件名」
        since / until: 时间范围（"YYYY-MM-DD" 或 "YYYY-MM-DD HH:MM:SS"），闭区间

    Returns: {"total": 命中条数, "limit":, "offset":, "tasks": [...]}
    """
    try:
        limit = max(1, min(int(limit or 50), 500))
    except (TypeError, ValueError):
        limit = 50
    try:
        offset = max(0, int(offset or 0))
    except (TypeError, ValueError):
        offset = 0

    where, args = [], []
    category = (category or "").strip()
    if category in ("image", "audio", "video"):
        where.append("url LIKE ?")
        args.append(category + "/%")
    keyword = (keyword or "").strip()
    if keyword:
        where.append("url LIKE ?")
        args.append("%" + keyword + "%")
    since = (since or "").strip()
    if since:
        where.append("created_at >= ?")
        args.append(since)
    until = (until or "").strip()
    if until:
        where.append("created_at <= ?")
        args.append(until + (" 23:59:59" if len(until) == 10 else ""))
    sql_where = (" WHERE " + " AND ".join(where)) if where else ""

    conn = _get_conn()
    total = conn.execute("SELECT COUNT(*) FROM completed_tasks" + sql_where, args).fetchone()[0]
    cur = conn.execute(
        "SELECT prompt_id, workflow_id, url, created_at FROM completed_tasks" + sql_where +
        " ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
        args + [limit, offset],
    )
    tasks = [{
        "prompt_id": row[0],
        "workflow_id": row[1] or "",
        "url": row[2] or "",
        "created_at": row[3] or "",
    } for row in cur.fetchall()]
    return {"total": total, "limit": limit, "offset": offset, "tasks": tasks}


def _fxai_root():
    return os.path.abspath(os.path.join(folder_paths.base_path, "fxai"))


def _split_url(url):
    """令牌 "image/子目录|a.png,b.png" -> (category, subdir, [文件名])"""
    left, _, files = (url or "").partition("|")
    category, _, subdir = left.partition("/")
    names = [f.strip() for f in files.split(",") if f.strip()]
    return category, subdir, names


def _task_path(category, subdir, name):
    """解析产物绝对路径；越界或类别不支持返回 None（与 /fxai/down/file 同一套防护）。"""
    if category not in ("image", "audio", "video"):
        return None
    root = _fxai_root()
    full = os.path.abspath(os.path.join(root, category, subdir, os.path.basename(name)))
    if not full.startswith(root + os.sep):
        return None
    return full


def clean_tasks(dry_run=False):
    """清理失效记录：产物在磁盘上已经不存在的，把整条记录删掉。

    部分文件还在的，只把 url 里已消失的文件名剔掉（记录留着，点开不会 404）。
    解析不出类别/文件名的记录一律不动——宁可留着，也不误删。
    """
    conn = _get_conn()
    rows = conn.execute("SELECT prompt_id, url FROM completed_tasks").fetchall()
    drop, trim = [], []
    for pid, url in rows:
        category, subdir, names = _split_url(url)
        if not names or category not in ("image", "audio", "video"):
            continue
        alive = []
        for n in names:
            p = _task_path(category, subdir, n)
            if p and os.path.isfile(p):
                alive.append(n)
        if not alive:
            drop.append(pid)
        elif len(alive) != len(names):
            trim.append((category + ("/" + subdir if subdir else "") + "|" + ",".join(alive), pid))
    if not dry_run:
        if drop:
            conn.executemany("DELETE FROM completed_tasks WHERE prompt_id=?", [(p,) for p in drop])
        if trim:
            conn.executemany("UPDATE completed_tasks SET url=? WHERE prompt_id=?", trim)
        conn.commit()
    return {
        "checked": len(rows),
        "removed": len(drop),
        "trimmed": len(trim),
        "kept": len(rows) - len(drop),
        "dry_run": bool(dry_run),
    }


def delete_task(prompt_id):
    """删除一条任务记录，连同它的全部产物文件。"""
    if not prompt_id:
        return {"ok": False, "error": "缺少 prompt_id"}
    conn = _get_conn()
    row = conn.execute("SELECT url FROM completed_tasks WHERE prompt_id=?", (prompt_id,)).fetchone()
    if row is None:
        return {"ok": False, "error": "记录不存在"}
    category, subdir, names = _split_url(row[0] or "")
    deleted, failed = [], []
    for n in names:
        p = _task_path(category, subdir, n)
        if not p:
            failed.append(n)
            continue
        if not os.path.isfile(p):
            continue                                   # 本来就没有，不算失败
        try:
            os.remove(p)
            deleted.append(n)
        except OSError:
            failed.append(n)
    conn.execute("DELETE FROM completed_tasks WHERE prompt_id=?", (prompt_id,))
    conn.commit()
    return {"ok": True, "prompt_id": prompt_id, "deleted_files": len(deleted), "failed": failed}


async def handle_query_task(request):
    if request.method == "POST":
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "需要 JSON body"}, status=400)
    else:
        data = request.query
    prompt_ids = data.get("prompt_ids", "") or data.get("prompt_id", "")
    if not prompt_ids:
        return web.json_response({"error": "缺少 prompt_ids"}, status=400)
    return web.json_response(get_tasks(prompt_ids))

# 路由已统一注册在 fxai_api_utils.py
