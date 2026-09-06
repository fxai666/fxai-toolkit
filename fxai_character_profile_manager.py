import json
import os
import sqlite3
import threading
import folder_paths
import server
from aiohttp import web

# ==============================================
# 短剧角色资源管理器：角色画像（形象照 + 音色 + 描述）持久化到 fxai.db
# ==============================================

DB_DIR = os.path.join(folder_paths.base_path, "fxai")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "fxai.db")
_local = threading.local()

# 角色字段（精简：形象照单选、音色、描述，与界面完全对应）
CHARACTER_FIELDS = [
    "name", "avatar", "voice", "description"
]


def _get_conn():
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH)
        _local.conn.execute("PRAGMA journal_mode=DELETE")
        _local.conn.execute("""
            CREATE TABLE IF NOT EXISTS character_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT DEFAULT NULL,
                avatar TEXT DEFAULT '',
                voice TEXT DEFAULT '',
                description TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now','localtime')),
                updated_at TEXT DEFAULT (datetime('now','localtime'))
            )
        """)
        # 迁移：新表 name 允许为空；若旧表 name 带 NOT NULL/UNIQUE 行约束，重建为可空 + 唯一索引(NULL 不冲突)
        cols = {r[1]: r for r in _local.conn.execute("PRAGMA table_info(character_profiles)").fetchall()}
        is_old_schema = ("description" not in cols) or (cols.get("name") and cols["name"][3])
        if is_old_schema:
            _local.conn.execute("""
                CREATE TABLE IF NOT EXISTS character_profiles_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT DEFAULT NULL,
                    avatar TEXT DEFAULT '',
                    voice TEXT DEFAULT '',
                    description TEXT DEFAULT '',
                    created_at TEXT DEFAULT (datetime('now','localtime')),
                    updated_at TEXT DEFAULT (datetime('now','localtime'))
                )
            """)
            try:
                _local.conn.execute("""
                    INSERT INTO character_profiles_new (name, avatar, voice, description, created_at, updated_at)
                    SELECT name, COALESCE(avatar,''), COALESCE(voice,''), COALESCE(description,''), created_at, updated_at
                    FROM character_profiles
                """)
                _local.conn.execute("DROP TABLE character_profiles")
                _local.conn.execute("ALTER TABLE character_profiles_new RENAME TO character_profiles")
                _local.conn.commit()
            except Exception as e:
                _local.conn.rollback()
                print(f"[凤希AI] character_profiles 迁移失败: {e}")
        _local.conn.execute("DROP INDEX IF EXISTS uq_char_name")
        _local.conn.commit()
    return _local.conn


def _row_to_dict(row):
    cols = ["id", "name", "avatar", "voice", "description", "created_at", "updated_at"]
    d = dict(zip(cols, row))
    return d


def list_characters():
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM character_profiles ORDER BY name").fetchall()
    return [_row_to_dict(r) for r in rows]


def get_character(name):
    conn = _get_conn()
    row = conn.execute("SELECT * FROM character_profiles WHERE name=?", (name,)).fetchone()
    return _row_to_dict(row) if row else None


def save_character(data):
    if not data:
        return None
    row_id = data.get("id") or None
    name = str(data.get("name", "") or "").strip() or None
    avatar = str(data.get("avatar", "") or "").strip()
    voice = str(data.get("voice", "") or "").strip()
    description = str(data.get("description", "") or "")
    conn = _get_conn()
    if row_id:
        # 有 id：更新
        conn.execute("""
            UPDATE character_profiles SET
                name=?, avatar=?, voice=?, description=?,
                updated_at=datetime('now','localtime')
            WHERE id=?
        """, (name, avatar, voice, description, row_id))
        conn.commit()
        if conn.total_changes > 0:
            return get_character_by_id(row_id)
        return None
    # 无 id：新增
    cur = conn.execute("""
        INSERT INTO character_profiles (name, avatar, voice, description)
        VALUES (?, ?, ?, ?)
    """, (name, avatar, voice, description))
    conn.commit()
    return get_character_by_id(cur.lastrowid)


def get_character_by_id(row_id):
    conn = _get_conn()
    row = conn.execute("SELECT * FROM character_profiles WHERE id=?", (row_id,)).fetchone()
    return _row_to_dict(row) if row else None


def get_characters_by_avatars(avatars):
    conn = _get_conn()
    result = {}
    if not avatars:
        return result
    placeholders = ",".join(["?"] * len(avatars))
    rows = conn.execute(
        f"SELECT * FROM character_profiles WHERE avatar IN ({placeholders})",
        list(avatars),
    ).fetchall()
    seen = set()
    for r in rows:
        d = _row_to_dict(r)
        av = d.get("avatar") or ""
        if av and av not in seen:
            seen.add(av)
            result[av] = d
    return result


def delete_character(row_id):
    conn = _get_conn()
    conn.execute("DELETE FROM character_profiles WHERE id=?", (row_id,))
    conn.commit()
    return True


def save_characters_batch(items):
    if not items:
        return []
    saved_items = []
    for data in items:
        item = save_character(data)
        if item:
            saved_items.append(item)
    return saved_items


# ==============================================
# REST API
# ==============================================
async def api_list_characters(request):
    return web.json_response({"characters": list_characters()})


async def api_save_character(request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "需要 JSON body"}, status=400)
    item = save_character(data)
    if not item:
        return web.json_response({"error": "保存失败"}, status=400)
    return web.json_response({"success": True, "character": item})


async def api_save_characters_batch(request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "需要 JSON body"}, status=400)
    if not isinstance(data, list):
        return web.json_response({"error": "body 必须是一个角色对象数组"}, status=400)
    saved = save_characters_batch(data)
    return web.json_response({"success": True, "saved": saved})


async def api_delete_character(request):
    try:
        data = await request.json()
    except Exception:
        data = request.query
    row_id = data.get("id") or data.get("name")
    if not row_id:
        return web.json_response({"error": "缺少 id"}, status=400)
    delete_character(row_id)
    return web.json_response({"success": True})


async def api_characters_by_avatars(request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "需要 JSON body"}, status=400)
    avatars = data.get("avatars", [])
    if not isinstance(avatars, list):
        return web.json_response({"error": "avatars 必须是数组"}, status=400)
    avatars = [str(x) for x in avatars if str(x).strip()]
    return web.json_response({"characters": get_characters_by_avatars(avatars)})


# 路由已统一注册在 fxai_api_utils.py

# ==============================================
# 节点：短剧角色资源管理器（行内弹窗式编辑，只存库不输出）
# ==============================================
class FxAiCharacterProfileManager:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "角色数据": ("STRING", {"default": "[]", "multiline": False}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "凤希AI/角色"

    def run(self, 角色数据=""):
        return ()