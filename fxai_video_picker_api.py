# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权

"""视频加载节点的可视化选片接口
================================

给 `FxAiVideoLoad` 用的三个只读接口，前端弹窗（js/fxai_video_load_picker.js）靠它们渲染：

    GET /fxai/video/browse?dir=<绝对目录或相对 input 的路径>
        列目录：子目录 + 视频（带 index，和节点内部排序完全一致）

    GET /fxai/video/thumb?path=<绝对文件>&t=<时间戳>
        视频首帧缩略图（ffmpeg 抽一帧，落 .thumbs/ 缓存）

    GET /fxai/video/file?path=<绝对文件>
        原视频流（给 <video> 播放/悬停预览，FileResponse 自带 range 支持）

安全：所有路径必须落在 ComfyUI 根目录之内，越界一律 403。
"""

import os
import asyncio
import hashlib
import subprocess

from aiohttp import web

try:
    import folder_paths
except Exception:                                    # 脱离 ComfyUI 单独 import 时
    folder_paths = None

try:
    from fxai_sort_utils import natural_key as _natural_key
except Exception:                                    # 脱离包环境独立运行时的兜底
    def _natural_key(name):
        return str(name).lower()

try:
    from server import PromptServer
except Exception:
    PromptServer = None


# 必须和 fxai_video_load.py 的 VIDEO_EXTENSIONS 一模一样，
# 否则弹窗里算出来的 视频索引 和节点实际读到的文件会对不上。
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpeg', '.mpg', '.webm')

THUMB_WIDTH = 320
THUMB_QUALITY = 5
FFMPEG_TIMEOUT = 40


# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------
def _base_dir():
    if folder_paths is None:
        return ""
    return os.path.abspath(folder_paths.base_path)


def _inside(path, base):
    """path 是否在 base 之内（Windows 下大小写不敏感）。"""
    try:
        a = os.path.normcase(os.path.abspath(path))
        b = os.path.normcase(os.path.abspath(base))
        return os.path.commonpath([a, b]) == b
    except Exception:
        return False


def _safe_dir(raw):
    """把前端传来的目录夹到 ComfyUI 根目录之内；空值给默认视频库。"""
    base = _base_dir()
    if not base:
        return None
    text = str(raw or "").strip().strip('"')
    if not text:
        d = os.path.join(base, "fxai", "video")
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass
        return d
    path = os.path.abspath(text)
    return path if _inside(path, base) else None


def _resolve_dir(raw):
    """选片窗口的起始目录。

    分段器把「视频文件」存成 input 下的相对路径，整个值扔过来时 abspath 会按
    进程工作目录解析成 I:\\ComfyUI\\xxx.mp4，弹窗就只剩一句「目录不存在」。
    所以相对路径按 input 目录展开，指到具体文件就退一级。

    / 开头的是「相对 ComfyUI 根」的写法（选片窗口的「用相对路径」给的就是这个），
    要按 base 展开；Windows 上 os.path.isabs("/x") 是 True，这个分支必须放前面。
    """
    text = str(raw or "").strip().strip('"')
    if text.startswith(("/", "\\")):
        text = os.path.join(_base_dir(), text.lstrip("/\\"))
    elif text and not os.path.isabs(text) and folder_paths is not None:
        text = os.path.join(folder_paths.get_input_directory(), text)
    if text and os.path.isfile(text):
        text = os.path.dirname(text)
    return _safe_dir(text)


def _safe_file(raw):
    """校验一个具体文件：必须在根目录内、必须是视频扩展名。"""
    path = _safe_dir(raw)
    if not path or not os.path.isfile(path):
        return None
    if os.path.splitext(path)[1].lower() not in VIDEO_EXTS:
        return None
    return path


def _roots():
    """弹窗顶部的快捷入口。"""
    base = _base_dir()
    out = []
    if not base:
        return out
    video_lib = os.path.join(base, "fxai", "video")
    if os.path.isdir(video_lib):
        out.append({"label": "视频库", "path": video_lib})
    if folder_paths is not None:
        for label, getter in (("input", "get_input_directory"), ("output", "get_output_directory")):
            try:
                d = os.path.abspath(getattr(folder_paths, getter)())
            except Exception:
                continue
            if os.path.isdir(d):
                out.append({"label": label, "path": d})
    return out


# ----------------------------------------------------------------------------
# 列目录
# ----------------------------------------------------------------------------
def _rel_path(full, base):
    """相对 ComfyUI 根的写法，前导 /，如 /fxai/video/ad/a.mp4。

    分段器的 resolve_video_path 认这个形式（/ 开头按 base_path 展开），
    存进工作流后换机器、换盘符都不用改。
    """
    try:
        rel = os.path.relpath(full, base).replace("\\", "/")
    except ValueError:
        return ""
    return "" if rel.startswith("..") else "/" + rel


def _list_videos(target):
    """排序规则必须和 FxAiVideoLoad.load_video 里的 sorted(...) 完全一致。"""
    if not os.path.isdir(target):
        return []
    try:
        names = sorted([f for f in os.listdir(target) if f.lower().endswith(VIDEO_EXTS)], key=_natural_key)
    except Exception:
        return []

    base = _base_dir()
    videos = []
    for idx, name in enumerate(names):
        full = os.path.join(target, name)
        try:
            st = os.stat(full)
            size, mtime = st.st_size, int(st.st_mtime)
        except Exception:
            size, mtime = 0, 0
        videos.append({
            "name": name,
            "path": full,
            "rel": _rel_path(full, base),
            "index": idx,          # 直接对应节点的「视频索引」
            "size": size,
            "mtime": mtime,
        })
    return videos


def _list_dirs(target):
    if not os.path.isdir(target):
        return []
    dirs = []
    try:
        for name in sorted(os.listdir(target), key=_natural_key):
            full = os.path.join(target, name)
            if name.startswith('.') or not os.path.isdir(full):
                continue
            try:
                count = len([f for f in os.listdir(full) if f.lower().endswith(VIDEO_EXTS)])
            except Exception:
                count = 0
            dirs.append({"name": name, "path": full, "videos": count})
    except Exception:
        return []
    return dirs


# ----------------------------------------------------------------------------
# 缩略图
# ----------------------------------------------------------------------------
def _thumb_dir():
    base = _base_dir()
    d = os.path.join(base, "fxai", "video", ".thumbs")
    os.makedirs(d, exist_ok=True)
    return d


def _thumb_path(video_path):
    key = hashlib.md5(os.path.normcase(video_path).encode("utf-8")).hexdigest()[:16]
    return os.path.join(_thumb_dir(), key + ".jpg")


def _make_thumb(video_path, out_path):
    """抽首帧。0.5s 处通常已经有画面（0s 可能是黑场），抽不到就退回 0s。"""
    for seek in ("0.5", "0"):
        cmd = [
            "ffmpeg", "-y", "-ss", seek, "-i", video_path,
            "-frames:v", "1",
            "-vf", f"scale={THUMB_WIDTH}:-2",
            "-q:v", str(THUMB_QUALITY),
            "-hide_banner", "-loglevel", "error",
            out_path,
        ]
        try:
            subprocess.run(cmd, timeout=FFMPEG_TIMEOUT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            continue
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            return True
    return False


# ----------------------------------------------------------------------------
# 路由
# ----------------------------------------------------------------------------
async def video_browse(request):
    target = _resolve_dir(request.query.get("dir", ""))
    if target is None:
        return web.json_response({"error": "目录不在 ComfyUI 根目录内"}, status=403)
    if not os.path.isdir(target):
        return web.json_response({"error": f"目录不存在：{target}"}, status=404)

    base = _base_dir()
    parent = os.path.dirname(target)
    if not _inside(parent, base) or os.path.normcase(parent) == os.path.normcase(target):
        parent = ""

    return web.json_response({
        "dir": target,
        "parent": parent,
        "roots": _roots(),
        "dirs": _list_dirs(target),
        "videos": _list_videos(target),
    })


async def video_thumb(request):
    video_path = _safe_file(request.query.get("path", ""))
    if not video_path:
        return web.json_response({"error": "文件不存在或不是视频"}, status=404)

    out = _thumb_path(video_path)
    try:
        stale = (not os.path.isfile(out)
                 or os.path.getmtime(out) < os.path.getmtime(video_path))
    except Exception:
        stale = True

    if stale:
        loop = asyncio.get_event_loop()
        ok = await loop.run_in_executor(None, _make_thumb, video_path, out)
        if not ok:
            return web.json_response({"error": "抽帧失败（文件可能损坏或没有视频流）"}, status=404)

    return web.FileResponse(out, headers={"Cache-Control": "max-age=600"})


async def video_file(request):
    video_path = _safe_file(request.query.get("path", ""))
    if not video_path:
        return web.json_response({"error": "文件不存在或不是视频"}, status=404)
    # FileResponse 自带 range 支持，<video> 才能拖动进度条
    return web.FileResponse(video_path, headers={"Cache-Control": "no-store"})


# ----------------------------------------------------------------------------
# 注册
# ----------------------------------------------------------------------------
def register():
    routes = getattr(getattr(PromptServer, "instance", None), "routes", None)
    if routes is None:
        return False
    routes.get("/fxai/video/browse")(video_browse)
    routes.get("/fxai/video/thumb")(video_thumb)
    routes.get("/fxai/video/file")(video_file)
    return True


try:
    register()
except Exception as _exc:                            # 不阻断插件加载
    print(f"[凤希AI] 视频选片路由注册失败：{_exc}")
