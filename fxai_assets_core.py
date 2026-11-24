# -*- coding: utf-8 -*-
"""
凤希AI 素材库 · 核心逻辑（纯函数，不依赖 HTTP 框架）
====================================================
被 fxai_assets_ui.py（挂在 ComfyUI 8188 上的 /fxweb）调用。

只负责：路径安全、列目录、搜索、统计、缩略图、回收站。
上传 / 新建文件夹不在这里做 —— 直接复用 fxai_api_utils 里已有的接口。

目录约定：
    <fxai>/                   素材根目录（audio / image / loras / prompts / video）
    <fxai>/fxai.db            任务记录 + 角色档案（受保护：不进列表，禁止读写/改名/删除）
    <fxai>/../fxweb/thumbs/   缩略图缓存（页面自己的缓存目录）
    <fxai>/../fxweb/trash/    回收站
"""

import hashlib
import json
import mimetypes
import os

try:
    from fxai_sort_utils import natural_key as _natural_key
except ImportError:   # 兜底：__init__ 还没把本目录注入 sys.path 时自愈，避免整个包加载失败
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fxai_sort_utils import natural_key as _natural_key
import posixpath
import re
import shutil
import time
import uuid
from datetime import datetime

ROOT = ""
DATA_DIR = ""
WEB_DATA_DIRNAME = "fxweb"      # 页面私有缓存目录（素材目录的同级），只放缩略图/回收站
THUMB_SUBDIR = "thumbs"         # fxweb/thumbs
TRASH_SUBDIR = "trash"          # fxweb/trash
THUMB_DIRNAME = ".fxai_thumbs"  # 旧位置（仅用于跳过列表与识别残留）
TRASH_DIRNAME = ".fxai_trash"
HIDDEN_SKIP = {THUMB_DIRNAME, TRASH_DIRNAME, "__pycache__", ".git", ".ipynb_checkpoints"}

# 受保护文件：不进列表、不可预览/下载/改名/移动/删除（resolve 层直接拦截）。
# 放在素材库根目录下的数据库及其附属文件，误删会丢数据。
PROTECTED_FILES = {"fxai.db", "fxai.db-wal", "fxai.db-shm", "fxai.db-journal"}


def is_protected(name):
    return name.lower() in PROTECTED_FILES

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".avif", ".tif", ".tiff", ".ico"}
VIDEO_EXT = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v", ".flv", ".wmv", ".mpg", ".mpeg", ".ts"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus", ".wma", ".aiff"}
TEXT_EXT = {".txt", ".md", ".json", ".csv", ".tsv", ".srt", ".ass", ".yaml", ".yml",
            ".log", ".ini", ".conf", ".xml", ".html", ".py", ".js", ".sh"}

MIME_OVERRIDE = {
    ".webp": "image/webp", ".avif": "image/avif", ".m4a": "audio/mp4",
    ".opus": "audio/ogg", ".mkv": "video/x-matroska", ".flac": "audio/flac",
    ".srt": "text/plain; charset=utf-8", ".ass": "text/plain; charset=utf-8",
    ".md": "text/plain; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".yaml": "text/plain; charset=utf-8", ".yml": "text/plain; charset=utf-8",
}

STAT_CACHE = {"ts": 0.0, "data": None}


def set_root(root, data_dir=None):
    """root = 素材目录；data_dir = 页面缓存（缩略图/回收站），默认在素材目录同级的 fxweb/。"""
    global ROOT, DATA_DIR
    ROOT = os.path.abspath(root)
    DATA_DIR = os.path.abspath(data_dir) if data_dir else os.path.join(
        os.path.dirname(ROOT), WEB_DATA_DIRNAME)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(thumb_dir(), exist_ok=True)
    os.makedirs(trash_dir(), exist_ok=True)
    STAT_CACHE["data"] = None
    return ROOT


def data_dir():
    return DATA_DIR


def thumb_dir():
    return os.path.join(DATA_DIR, THUMB_SUBDIR)


def trash_dir():
    return os.path.join(DATA_DIR, TRASH_SUBDIR)


# ------------------------------------------------------------------ 基础
def hsize(n):
    try:
        n = float(n)
    except Exception:
        return "-"
    if n < 1024:
        return "%d B" % n
    for unit in ("KB", "MB", "GB", "TB"):
        n /= 1024.0
        if n < 1024 or unit == "TB":
            return "%.1f %s" % (n, unit)
    return "%.1f TB" % n


def kind_of(name):
    ext = os.path.splitext(name)[1].lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in TEXT_EXT:
        return "text"
    return "other"


def mime_of(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in MIME_OVERRIDE:
        return MIME_OVERRIDE[ext]
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


def clean_rel(rel):
    """规范化成 a/b/c，拒绝 .. 与盘符。"""
    rel = (rel or "").replace("\\", "/").strip().lstrip("/")
    parts = []
    for seg in rel.split("/"):
        seg = seg.strip()
        if seg in ("", "."):
            continue
        if seg == "..":
            raise ValueError("路径不允许包含 ..")
        if "\x00" in seg or ":" in seg:
            raise ValueError("路径含非法字符")
        parts.append(seg)
    return "/".join(parts)


def resolve(rel):
    """相对路径 → 绝对路径。

    只校验「字面路径」不逃出 ROOT（拦 `..`、盘符、非法字符），**不再用 realpath 拦软链**。

    原因：素材库就是为了资源共享，image/pose、image/clothes 这类目录经常是指向别的盘的
    符号链接 / 目录联接（例如 /i/posegallery/repo/img）。原来的 realpath 校验会把它们
    全部判成「路径越界」，点进去就报错，等于把挂进来的素材全废了。
    链接本身是 ROOT 内的一个条目，进不进得去由文件系统决定，不由字符串强度决定。
    """
    rel = clean_rel(rel)
    target = os.path.abspath(os.path.join(ROOT, *rel.split("/"))) if rel else ROOT
    base = os.path.normcase(os.path.abspath(ROOT))
    cand = os.path.normcase(target)
    if cand != base and not cand.startswith(base + os.sep):
        raise ValueError("路径越界")
    if is_protected(os.path.basename(target)):
        raise ValueError("该文件受保护，不能通过素材库操作")
    return target


def rel_of(abs_path):
    rel = os.path.relpath(abs_path, ROOT)
    return "" if rel == "." else rel.replace("\\", "/")


def walk_tree(top):
    """遍历目录树，**跟随符号链接 / 目录联接**，语义同 os.walk（可改 dirnames 剪枝）。

    为什么不用 os.walk(top, followlinks=True)：外挂素材目录之间可能互指（pose 与
    pose/costume 挂同一处），一路跟下去会绕圈。这里用 realpath 去重，同一个真实目录
    只进一次，既能看到外挂内容，又不会死循环。
    """
    seen = set()

    def rec(cur):
        try:
            real = os.path.normcase(os.path.realpath(cur))
        except OSError:
            real = os.path.normcase(os.path.abspath(cur))
        if real in seen:
            return
        seen.add(real)
        dirs, files = [], []
        try:
            with os.scandir(cur) as it:
                for e in it:
                    name = e.name
                    if name.startswith(".") or is_protected(name):
                        continue
                    try:
                        isdir = e.is_dir(follow_symlinks=True)
                    except OSError:
                        isdir = False
                    (dirs if isdir else files).append(name)
        except OSError:
            return
        dirs.sort()
        yield cur, dirs, files          # 调用方可在 yield 期间原地改 dirs 来剪枝
        for d in dirs:
            for item in rec(os.path.join(cur, d)):
                yield item

    for item in rec(top):
        yield item


def unique_path(target):
    if not os.path.exists(target):
        return target
    base, ext = os.path.splitext(target)
    i = 1
    while True:
        cand = "%s (%d)%s" % (base, i, ext)
        if not os.path.exists(cand):
            return cand
        i += 1


def stamp(ts):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


# ------------------------------------------------------------------ 列目录
def crumbs_of(rel):
    """相对路径 → 面包屑。列表和搜索都要用，别只在列表里拼。"""
    crumbs = [{"name": "素材根目录", "path": ""}]
    acc = []
    for seg in (rel.split("/") if rel else []):
        acc.append(seg)
        crumbs.append({"name": seg, "path": "/".join(acc)})
    return crumbs


def dir_counters(abs_dir):
    files = subdirs = 0
    try:
        with os.scandir(abs_dir) as it:
            for e in it:
                if e.name.startswith(".") or is_protected(e.name):
                    continue
                try:
                    # 跟随软链：挂进来的素材目录（符号链接 / 目录联接）也要算进数量
                    if e.is_dir(follow_symlinks=True):
                        subdirs += 1
                    else:
                        files += 1
                except OSError:
                    continue
    except OSError:
        pass
    return files, subdirs


def file_item(full, rel_path):
    st = os.stat(full)
    name = os.path.basename(full)
    return {
        "name": name, "path": rel_path, "is_dir": False, "kind": kind_of(name),
        "ext": os.path.splitext(name)[1].lower().lstrip("."),
        "size": st.st_size, "size_str": hsize(st.st_size),
        "mtime": int(st.st_mtime), "mtime_str": stamp(st.st_mtime),
    }


def list_dir(rel, keyword="", kind="", sort="name", order="asc", page=1, size=60):
    abs_dir = resolve(rel)
    if not os.path.isdir(abs_dir):
        raise ValueError("目录不存在")

    dirs, files = [], []
    with os.scandir(abs_dir) as it:
        for e in it:
            name = e.name
            if name.startswith(".") or is_protected(name):
                continue
            if rel == "" and name in HIDDEN_SKIP:
                continue
            try:
                st = e.stat(follow_symlinks=True)
            except OSError:
                # 链接指向的盘掉线时，至少把条目列出来，别让整条目录消失
                try:
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    continue
            rel_path = posixpath.join(rel, name) if rel else name
            try:
                is_dir = e.is_dir(follow_symlinks=True)   # 软链/目录联接也算目录
            except OSError:
                is_dir = False
            if is_dir:
                nf, nd = dir_counters(e.path)
                dirs.append({
                    "name": name, "path": rel_path, "is_dir": True,
                    "mtime": int(st.st_mtime), "mtime_str": stamp(st.st_mtime),
                    "files": nf, "subdirs": nd, "count": nf + nd,
                })
            else:
                files.append(file_item(e.path, rel_path))

    kw = (keyword or "").strip().lower()
    if kw:
        dirs = [d for d in dirs if kw in d["name"].lower()]
        files = [f for f in files if kw in f["name"].lower()]
    if kind:
        # 类型筛选只作用于「文件」：文件夹永远保留。
        # 素材都是放在子目录里的，筛类型时把文件夹一起藏掉，页面会变成空白，看不出东西在哪。
        wanted = {x for x in kind.split(",") if x and x != "folder"}
        if wanted:
            files = [f for f in files if f["kind"] in wanted]
        else:
            files = []

    rev = (order == "desc")
    if sort == "size":
        files.sort(key=lambda f: f["size"], reverse=rev)
    elif sort == "time":
        files.sort(key=lambda f: f["mtime"], reverse=rev)
    else:
        files.sort(key=lambda f: _natural_key(f["name"]), reverse=rev)
    dirs.sort(key=lambda d: _natural_key(d["name"]))

    total = len(files)
    sized = sum(f["size"] for f in files)
    page_size = max(1, min(int(size or 60), 500))
    page = max(1, int(page or 1))
    pages = max(1, (total + page_size - 1) // page_size)
    page = min(page, pages)
    start = (page - 1) * page_size

    return {
        "path": rel, "crumbs": crumbs_of(rel), "dirs": dirs,
        "files": files[start:start + page_size],
        "total": total, "total_size": sized, "total_size_str": hsize(sized),
        "page": page, "pages": pages, "size": page_size, "dir_count": len(dirs),
    }


def search_all(rel, keyword, kind="", limit=800):
    kw = (keyword or "").strip().lower()
    if not kw:
        return {"files": [], "dirs": [], "truncated": False}
    base = resolve(rel)
    hits_files, hits_dirs = [], []
    truncated = False
    base_depth = rel.count("/") + (1 if rel else 0)

    for dirpath, dirnames, filenames in walk_tree(base):
        dirnames[:] = [d for d in dirnames if d not in HIDDEN_SKIP]
        cur = rel_of(dirpath)
        if (cur.count("/") + 1 if cur else 0) - base_depth > 6:
            dirnames[:] = []
            continue
        for d in dirnames:
            if kw in d.lower():
                full = os.path.join(dirpath, d)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                nf, nd = dir_counters(full)
                hits_dirs.append({
                    "name": d, "path": rel_of(full), "is_dir": True,
                    "mtime": int(st.st_mtime), "mtime_str": stamp(st.st_mtime),
                    "files": nf, "subdirs": nd, "count": nf + nd,
                })
        for fn in filenames:
            if fn.startswith(".") or is_protected(fn) or kw not in fn.lower():
                continue
            full = os.path.join(dirpath, fn)
            try:
                item = file_item(full, rel_of(full))
            except OSError:
                continue
            if kind and item["kind"] not in {x for x in kind.split(",") if x}:
                continue
            hits_files.append(item)
            if len(hits_files) + len(hits_dirs) >= limit:
                truncated = True
                break
        if truncated:
            break

    hits_files.sort(key=lambda f: f["mtime"], reverse=True)
    hits_dirs.sort(key=lambda d: d["name"].lower())
    return {"crumbs": crumbs_of(rel), "files": hits_files, "dirs": hits_dirs, "truncated": truncated}


def global_stat(force=False):
    now = time.time()
    if not force and STAT_CACHE["data"] and now - STAT_CACHE["ts"] < 60:
        return STAT_CACHE["data"]

    out = {"total_files": 0, "total_size": 0, "by_kind": {}, "by_top": {}, "updated": int(now)}
    for k in ("image", "video", "audio", "text", "other"):
        out["by_kind"][k] = {"count": 0, "size": 0}

    for dirpath, dirnames, filenames in walk_tree(ROOT):
        dirnames[:] = [d for d in dirnames if d not in HIDDEN_SKIP]
        cur = rel_of(dirpath)
        top = cur.split("/")[0] if cur else ""
        if top:
            # 分类卡片要讲清「里面还有子目录」，否则只看得到一个总文件数，不知道往里点
            slot = out["by_top"].setdefault(top, {"count": 0, "size": 0, "dirs": 0, "samples": []})
            slot["dirs"] += len(dirnames)
            if cur == top:
                for d in dirnames[:8]:
                    slot["samples"].append(d)
        for fn in filenames:
            if fn.startswith(".") or is_protected(fn):
                continue
            try:
                size = os.path.getsize(os.path.join(dirpath, fn))
            except OSError:
                continue
            k = kind_of(fn)
            out["total_files"] += 1
            out["total_size"] += size
            out["by_kind"][k]["count"] += 1
            out["by_kind"][k]["size"] += size
            if top:
                slot = out["by_top"].setdefault(top, {"count": 0, "size": 0, "dirs": 0, "samples": []})
                slot["count"] += 1
                slot["size"] += size

    out["total_size_str"] = hsize(out["total_size"])
    for k in out["by_kind"]:
        out["by_kind"][k]["size_str"] = hsize(out["by_kind"][k]["size"])
    for k in out["by_top"]:
        out["by_top"][k]["size_str"] = hsize(out["by_top"][k]["size"])
        out["by_top"][k]["samples"] = sorted(out["by_top"][k]["samples"])[:5]

    STAT_CACHE["ts"] = now
    STAT_CACHE["data"] = out
    return out


def clear_stat_cache():
    STAT_CACHE["data"] = None


# ------------------------------------------------------------------ 缩略图
def thumb_key(rel_path, mtime, width):
    raw = "%s|%d|%d" % (rel_path, int(mtime), width)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def thumb_path(rel_path, mtime, width):
    """缩略图固定存在 <fxai>/.fxai_thumbs/<hash>.webp，与素材目录完全隔离。"""
    return os.path.join(thumb_dir(), "%s.webp" % thumb_key(rel_path, mtime, width))


def make_thumb(rel_path, abs_path, width=480):
    """返回缩略图路径；没有 Pillow 或生成失败返回 None（调用方退回原图）。"""
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        width = max(120, min(int(width), 1600))
        mtime = os.path.getmtime(abs_path)
        out = thumb_path(rel_path, mtime, width)
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            return out
        os.makedirs(thumb_dir(), exist_ok=True)
        with Image.open(abs_path) as im:
            im = im.convert("RGB")
            if im.width > width:
                im = im.resize((width, max(1, int(im.height * width / im.width))), Image.LANCZOS)
            tmp = out + ".tmp"
            im.save(tmp, "WEBP", quality=82, method=4)
        os.replace(tmp, out)
        return out
    except Exception:
        return None


def prune_thumbs(max_files=20000, max_age_days=60):
    """缩略图缓存清理：超过 max_files 或太旧的就删掉，不会被当成素材。"""
    d = thumb_dir()
    if not os.path.isdir(d):
        return 0
    items = []
    now = time.time()
    removed = 0
    try:
        for name in os.listdir(d):
            full = os.path.join(d, name)
            if not os.path.isfile(full):
                continue
            try:
                st = os.stat(full)
            except OSError:
                continue
            if now - st.st_mtime > max_age_days * 86400:
                try:
                    os.remove(full)
                    removed += 1
                except OSError:
                    pass
                continue
            items.append((st.st_mtime, full))
    except OSError:
        return removed

    if len(items) > max_files:
        items.sort()
        for _, full in items[:len(items) - max_files]:
            try:
                os.remove(full)
                removed += 1
            except OSError:
                pass
    return removed


# ------------------------------------------------------------------ 回收站
def trash_put(abs_path):
    rel = rel_of(abs_path)
    name = os.path.basename(abs_path)
    if is_protected(name):
        raise ValueError("该文件受保护，不能删除")
    tid = "%s__%s" % (time.strftime("%Y%m%d-%H%M%S", time.localtime()), uuid.uuid4().hex[:6])
    bucket = os.path.join(trash_dir(), tid)
    os.makedirs(bucket, exist_ok=True)
    payload = os.path.join(bucket, name)
    shutil.move(abs_path, payload)

    size = nfiles = 0
    if os.path.isdir(payload):
        for dp, _dn, fns in os.walk(payload):
            for fn in fns:
                try:
                    size += os.path.getsize(os.path.join(dp, fn))
                    nfiles += 1
                except OSError:
                    pass
    else:
        try:
            size = os.path.getsize(payload)
            nfiles = 1
        except OSError:
            pass

    meta = {
        "id": tid, "name": name, "rel": rel,
        "deleted_at": int(time.time()), "deleted_str": stamp(time.time()),
        "is_dir": os.path.isdir(payload), "size": size,
        "size_str": hsize(size), "files": nfiles,
    }
    with open(os.path.join(bucket, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    return meta


def trash_list():
    base = trash_dir()
    if not os.path.isdir(base):
        return []
    items = []
    for name in os.listdir(base):
        bucket = os.path.join(base, name)
        mf = os.path.join(bucket, "meta.json")
        if not os.path.isfile(mf):
            continue
        try:
            with open(mf, encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            continue
        meta["id"] = meta.get("id", name)
        meta["exists"] = os.path.exists(os.path.join(bucket, meta.get("name", "")))
        meta.setdefault("size_str", hsize(meta.get("size", 0)))
        items.append(meta)
    items.sort(key=lambda m: m.get("deleted_at", 0), reverse=True)
    return items


def trash_restore(tid):
    bucket = os.path.join(trash_dir(), tid)
    mf = os.path.join(bucket, "meta.json")
    if not os.path.isfile(mf):
        raise ValueError("回收站条目不存在")
    with open(mf, encoding="utf-8") as fh:
        meta = json.load(fh)
    payload = os.path.join(bucket, meta["name"])
    if not os.path.exists(payload):
        raise ValueError("文件已不存在")
    dest = resolve(meta["rel"])
    os.makedirs(os.path.dirname(dest) or ROOT, exist_ok=True)
    dest = unique_path(dest)
    shutil.move(payload, dest)
    shutil.rmtree(bucket, ignore_errors=True)
    return rel_of(dest)


def trash_purge(tid):
    bucket = os.path.join(trash_dir(), tid)
    if not os.path.isdir(bucket):
        raise ValueError("回收站条目不存在")
    shutil.rmtree(bucket, ignore_errors=True)
    return tid


def trash_empty():
    n = 0
    for it in trash_list():
        try:
            trash_purge(it["id"])
            n += 1
        except Exception:
            pass
    return n
