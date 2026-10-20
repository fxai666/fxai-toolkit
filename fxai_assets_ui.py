# -*- coding: utf-8 -*-
"""
凤希AI 素材库 · ComfyUI 内置页面
================================
挂在 ComfyUI 自己的端口上（默认 8188），不额外起进程、不额外占端口：

    http://127.0.0.1:8188/fxweb

- **上传 / 新建文件夹**：直接复用 fxai_api_utils 已有的接口，这里不重复实现
    POST /fxai/image/upload   (field: image,  subdir)
    POST /fxai/audio/upload   (field: audio,  subdir)
    POST /fxai/video/upload   (field: video,  subdir)
    POST /fxai/folders/create (query: subdir, folder)
- **列表 / 缩略图 / 预览 / 删除 / 改名 / 移动 / 回收站**：由本文件提供。
  注意：这里刻意**不做**「删除后自动重编号」——素材库是文件管理语义，
  删一个不该让别的文件改名；要重排请用工作流里的 /fxai/image/apply。

缩略图统一存 `<fxweb>/thumbs/`，回收站存 `<fxweb>/trash/`（页面自己的缓存目录，
不放进素材目录，免得别的软件操作素材时把它们连带删掉）。
"""

import os
import shutil
import time
from urllib.parse import quote

from aiohttp import web

import folder_paths

try:
    from server import PromptServer
except Exception:  # 极少数情况下单独 import 进来做测试
    PromptServer = None

from . import fxai_assets_core as core

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "assets.html")
ROOT = os.path.join(folder_paths.base_path, "fxai")   # 素材目录（数据都在这里）
DATA = os.path.join(folder_paths.base_path, "fxweb")  # 页面缓存：缩略图 + 回收站

MAX_THUMB_BYTES = 40 * 1024 * 1024   # 超过 40MB 的图不做缩略图，直接给原图
MAX_TEXT_BYTES = 512 * 1024


def _init():
    try:
        os.makedirs(ROOT, exist_ok=True)
        core.set_root(ROOT, DATA)
        core.prune_thumbs()
    except Exception as exc:
        print("[凤希AI] 素材库初始化失败：%s" % exc)


_init()


# ------------------------------------------------------------------ 工具
def _cd(name, inline=False):
    """中文文件名下载不乱码。"""
    disp = "inline" if inline else "attachment"
    ascii_name = name.encode("ascii", "replace").decode("ascii").replace('"', "_")
    return "%s; filename=\"%s\"; filename*=UTF-8''%s" % (disp, ascii_name, quote(name, safe=""))


def _json(data, status=200, **kw):
    out = {"ok": True}
    out.update(data)
    out.update(kw)
    return web.json_response(out, status=status)


def _err(msg, status=400):
    return web.json_response({"ok": False, "error": str(msg)}, status=status)


def _q(request, key, default=""):
    return request.query.get(key, default)


async def _body(request):
    try:
        data = await request.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _abs(rel):
    return core.resolve(rel)


# ------------------------------------------------------------------ 页面
async def assets_page(request):
    if not os.path.isfile(PAGE):
        return web.Response(text="找不到素材库页面：%s" % PAGE, status=500, content_type="text/plain")
    with open(PAGE, "rb") as fh:
        body = fh.read()
    return web.Response(body=body, content_type="text/html", charset="utf-8",
                        headers={"Cache-Control": "no-store"})


# ------------------------------------------------------------------ 列表 / 统计
async def api_list(request):
    try:
        rel = _q(request, "path")
        keyword = _q(request, "q")
        if keyword:
            data = core.search_all(rel, keyword, _q(request, "type"))
            return _json({"path": core.clean_rel(rel), "search": keyword, **data})
        data = core.list_dir(
            rel, "", _q(request, "type"),
            _q(request, "sort", "name"), _q(request, "order", "asc"),
            _q(request, "page", "1"), _q(request, "size", "60"),
        )
        return _json(data)
    except ValueError as exc:
        return _err(exc, 400)
    except Exception as exc:
        print("[凤希AI] 素材库列表出错：%s" % exc)
        return _err("列表失败：%s" % exc, 500)


async def api_stat(request):
    try:
        return _json({"stat": core.global_stat(force=_q(request, "force") in ("1", "true"))})
    except Exception as exc:
        return _err("统计失败：%s" % exc, 500)


# ------------------------------------------------------------------ 文件输出
async def api_file(request):
    try:
        target = _abs(_q(request, "path"))
        if not os.path.isfile(target):
            return _err("文件不存在", 404)
        return web.FileResponse(target, headers={
            "Content-Type": core.mime_of(target),
            "Content-Disposition": _cd(os.path.basename(target), inline=True),
            "Cache-Control": "public, max-age=3600",
        })
    except ValueError as exc:
        return _err(exc, 400)


async def api_thumb(request):
    try:
        rel = core.clean_rel(_q(request, "path"))
        target = core.resolve(rel)
        if not os.path.isfile(target):
            return _err("文件不存在", 404)
        try:
            width = int(_q(request, "w", "480"))
        except ValueError:
            width = 480
        if os.path.getsize(target) > MAX_THUMB_BYTES:
            return web.FileResponse(target, headers={"Cache-Control": "public, max-age=3600"})
        thumb = core.make_thumb(rel, target, width)
        if thumb and os.path.isfile(thumb):
            return web.FileResponse(thumb, headers={
                "Content-Type": "image/webp",
                "Cache-Control": "public, max-age=86400",
            })
        return web.FileResponse(target, headers={
            "Content-Type": core.mime_of(target),
            "Cache-Control": "public, max-age=3600",
        })
    except ValueError as exc:
        return _err(exc, 400)


async def api_download(request):
    try:
        target = _abs(_q(request, "path"))
        if not os.path.isfile(target):
            return _err("文件不存在", 404)
        return web.FileResponse(target, headers={
            "Content-Type": core.mime_of(target),
            "Content-Disposition": _cd(os.path.basename(target), inline=False),
        })
    except ValueError as exc:
        return _err(exc, 400)


async def api_text(request):
    try:
        target = _abs(_q(request, "path"))
        if not os.path.isfile(target):
            return _err("文件不存在", 404)
        size = os.path.getsize(target)
        with open(target, "rb") as fh:
            chunk = fh.read(MAX_TEXT_BYTES)
        return _json({"text": chunk.decode("utf-8", "replace"),
                      "truncated": size > MAX_TEXT_BYTES,
                      "size": size})
    except ValueError as exc:
        return _err(exc, 400)


# ------------------------------------------------------------------ 修改
async def api_delete(request):
    body = await _body(request)
    paths = body.get("paths") or []
    trashed, failed = [], []
    for rel in paths:
        try:
            src = _abs(rel)
            if os.path.abspath(src) == os.path.abspath(core.ROOT):
                failed.append({"path": rel, "error": "不能删除根目录"})
                continue
            if not os.path.exists(src):
                failed.append({"path": rel, "error": "不存在"})
                continue
            trashed.append(core.trash_put(src))
        except Exception as exc:
            failed.append({"path": rel, "error": str(exc)})
    core.clear_stat_cache()
    return _json({"trashed": trashed, "failed": failed})


async def api_rename(request):
    body = await _body(request)
    rel = body.get("path") or ""
    name = (body.get("name") or "").strip()
    if not name:
        return _err("新名称不能为空")
    name = "".join(c for c in name if c not in '\\/:*?"<>|').strip()
    if not name:
        return _err("名称含非法字符")
    src = _abs(rel)
    if not os.path.exists(src):
        return _err("目标不存在", 404)
    if os.path.abspath(src) == os.path.abspath(core.ROOT):
        return _err("不能重命名根目录")
    if core.is_protected(name):
        return _err("该名称受保护，不能作为新名称")
    dst = os.path.join(os.path.dirname(src), name)
    if os.path.normcase(dst) != os.path.normcase(src) and os.path.exists(dst):
        return _err("同名文件已存在", 409)
    os.rename(src, dst)
    return _json({"path": core.rel_of(dst)})


async def api_move(request):
    body = await _body(request)
    dest_rel = body.get("dest") or ""
    paths = body.get("paths") or []
    dest_dir = _abs(dest_rel)
    os.makedirs(dest_dir, exist_ok=True)
    moved, failed = [], []
    for rel in paths:
        try:
            src = _abs(rel)
            if not os.path.exists(src):
                failed.append({"path": rel, "error": "不存在"})
                continue
            if os.path.normcase(os.path.dirname(src)) == os.path.normcase(dest_dir):
                failed.append({"path": rel, "error": "已在目标目录"})
                continue
            if os.path.isdir(src) and os.path.normcase(dest_dir).startswith(os.path.normcase(src) + os.sep):
                failed.append({"path": rel, "error": "不能移到自己里面"})
                continue
            final = core.unique_path(os.path.join(dest_dir, os.path.basename(src)))
            shutil.move(src, final)
            moved.append(core.rel_of(final))
        except Exception as exc:
            failed.append({"path": rel, "error": str(exc)})
    core.clear_stat_cache()
    return _json({"moved": moved, "failed": failed})


# ------------------------------------------------------------------ 回收站
async def api_trash(request):
    return _json({"items": core.trash_list()})


async def api_trash_restore(request):
    body = await _body(request)
    done, failed = [], []
    for tid in (body.get("ids") or []):
        try:
            done.append({"id": tid, "restored": core.trash_restore(tid)})
        except Exception as exc:
            failed.append({"id": tid, "error": str(exc)})
    core.clear_stat_cache()
    return _json({"results": done, "failed": failed})


async def api_trash_purge(request):
    body = await _body(request)
    done, failed = [], []
    for tid in (body.get("ids") or []):
        try:
            core.trash_purge(tid)
            done.append(tid)
        except Exception as exc:
            failed.append({"id": tid, "error": str(exc)})
    return _json({"purged": done, "failed": failed})


async def api_trash_empty(request):
    n = core.trash_empty()
    return _json({"count": n})


# ------------------------------------------------------------------ 注册
BASE = "/fxweb"      # 素材库唯一入口：/fxweb 直接打开页面，接口在 /fxweb/api/*


def register_routes():
    if PromptServer is None:
        print("[凤希AI] 素材库：没有 PromptServer，跳过路由注册")
        return
    r = PromptServer.instance.routes

    table = [
        ("get", "", assets_page),
        ("get", "/", assets_page),
        ("get", "/api/list", api_list),
        ("get", "/api/stat", api_stat),
        ("get", "/api/file", api_file),
        ("get", "/api/thumb", api_thumb),
        ("get", "/api/download", api_download),
        ("get", "/api/text", api_text),
        ("get", "/api/trash", api_trash),
        ("post", "/api/delete", api_delete),
        ("post", "/api/rename", api_rename),
        ("post", "/api/move", api_move),
        ("post", "/api/trash/restore", api_trash_restore),
        ("post", "/api/trash/purge", api_trash_purge),
        ("post", "/api/trash/empty", api_trash_empty),
    ]
    for method, suffix, handler in table:
        getattr(r, method)(BASE + suffix)(handler)

    print("[凤希AI] 素材库已挂载：%s  →  素材 %s" % (BASE, ROOT))
    print("[凤希AI]   缩略图：%s" % core.thumb_dir())
    print("[凤希AI]   回收站：%s" % core.trash_dir())


try:
    register_routes()
except Exception as _exc:
    print("[凤希AI] 素材库路由注册失败：%s" % _exc)
