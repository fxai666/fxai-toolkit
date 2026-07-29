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

def get_task(prompt_id):
    conn = _get_conn()
    cur = conn.execute("SELECT * FROM completed_tasks WHERE prompt_id=?", (prompt_id,))
    row = cur.fetchone()
    if not row:
        return None
    return {
        "prompt_id": row[0],
        "workflow_id": row[1],
        "url": row[2] or '',
        "created_at": row[3]
    }

async def handle_query_task(request):
    prompt_id = request.query.get("prompt_id", "")
    if not prompt_id:
        return web.json_response({"error": "缺少 prompt_id"}, status=400)
    task = get_task(prompt_id)
    if not task:
        return web.json_response({"error": "未找到"}, status=404)
    return web.json_response(task)

try:
    PromptServer.instance.routes.get("/fxai/tasks/result")(handle_query_task)
except Exception as e:
    print(f"[凤希AI任务存储] 路由注册失败：{e}")
