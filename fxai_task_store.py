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
    """自动获取当前运行中的 prompt_id"""
    try:
        running, _ = PromptServer.instance.prompt_queue.get_current_queue()
        if running:
            return running[0][1]
    except:
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
    if not prompt_id or not files:
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
