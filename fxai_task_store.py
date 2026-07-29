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
                files TEXT DEFAULT '[]',
                directory TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now','localtime'))
            )
        """)
        _local.conn.commit()
    return _local.conn

def save_task(prompt_id, workflow_id, files, directory):
    if not prompt_id:
        return
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO completed_tasks (prompt_id, workflow_id, files, directory) VALUES (?,?,?,?)",
        (prompt_id, workflow_id or '', files or '', directory)
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
        "files": row[2] or '',
        "directory": row[3],
        "created_at": row[4]
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
