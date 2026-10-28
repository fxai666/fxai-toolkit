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
