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

    POST /fxai/video/picker/upload?dir=<绝对目录>
        把视频传到当前浏览的目录（multipart 字段名 file）

⚠️ 上传**不能**复用 /fxai/video/upload：那条在 fxai_api_utils.py 里，语义是
「subdir 相对 fxai/video、自动改名成 001_xxx」，而且它先注册，同名路由抢不过来。
选片窗口要的是「传到我正在看的这个目录」（含 input / output），所以单独开一条。

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


def _to_abs(text):
    """把前端传来的写法统一成绝对路径。

    认三种写法：绝对路径、相对 input 的路径、以 / 开头的「相对 ComfyUI 根」写法。

    ⚠️ / 开头有两种可能，**必须先按绝对路径判存在性**：
    - Linux 上 `/root/ComfyUI/fxai/video` 是真绝对路径，先拼 base 会变成
      `/root/ComfyUI/root/ComfyUI/fxai/video`（弹窗里看到的就是这个鬼路径）；
    - Windows 上 `/fxai/video` 的语义是「相对根」，按绝对解释必然不存在，
      这时才回落到拼 base（`os.path.isabs("/x")` 在 Windows 上也是 True，不能靠它判断）。
    """
    base = _base_dir()
    cand = os.path.abspath(text)
    if os.path.exists(cand):
        return cand
    if text.startswith(("/", "\\")) and base:
        return os.path.join(base, text.lstrip("/\\"))
    if base and folder_paths is not None and not os.path.isabs(text):
        return os.path.join(folder_paths.get_input_directory(), text)
    return cand


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
    path = _to_abs(text)
    return path if _inside(path, base) else None


def _resolve_dir(raw):
    """选片窗口的起始目录。

    分段器把「视频文件」存成 input 下的相对路径、或 / 开头的相对根写法、或绝对路径，
    三种都可能扔过来。展开规则统一交给 _to_abs，指到具体文件就退一级。
    """
    text = str(raw or "").strip().strip('"')
    if text:
        text = _to_abs(text)
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


def _root_paths():
    return [r["path"] for r in _roots()]


def _under_roots(path):
    """path 是否落在某个快捷入口（视频库 / input / output）之内，含等于它自己。"""
    if not path:
        return False
    return any(_inside(path, root) for root in _root_paths())


def _parent_within_roots(target):
    """上一级目录，但**不许爬出** 视频库 / input / output 这三块。

    到顶（parent 等于某个入口本身）或再往上（落到 ComfyUI 根、fxai/ 之类）一律给空串，
    前端据此把「上一级」按钮灰掉。
    """
    parent = os.path.dirname(target)
    if not parent or os.path.normcase(parent) == os.path.normcase(target):
        return ""
    return parent if _under_roots(parent) else ""


def _default_dir():
    """兜底目录：优先视频库，其次第一个存在的入口。"""
    base = _base_dir()
    if not base:
        return ""
    video_lib = os.path.join(base, "fxai", "video")
    if os.path.isdir(video_lib):
        return video_lib
    for root in _root_paths():
        if os.path.isdir(root):
            return root
    return ""


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
    requested = str(request.query.get("dir", "") or "")
    target = _resolve_dir(requested)

    # 起始目录不存在（节点里存的是别的机器上的路径、目录被删、盘符变了）时
    # **不能直接报错**——前端拿不到 roots 就渲染不出任何可点的东西，弹窗变成死路。
    # 这里退回默认视频库，并把原始请求带回去让前端提示一句。
    fallback = False
    if target is None:
        return web.json_response({"error": "目录不在 ComfyUI 根目录内", "roots": _roots()}, status=403)
    if not os.path.isdir(target):
        alt = _default_dir()
        if not alt:
            return web.json_response(
                {"error": "目录不存在：%s" % target, "roots": _roots()}, status=404)
        target, fallback = alt, True

    return web.json_response({
        "dir": target,
        "parent": _parent_within_roots(target),
        "roots": _roots(),
        "dirs": _list_dirs(target),
        "videos": _list_videos(target),
        "fallback": fallback,
        "requested": requested,
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
# 上传
# ----------------------------------------------------------------------------
_BAD_NAME_CHARS = '<>:"/\\|?*'


def _safe_name(stem):
    """去掉文件名里的路径分隔符和 Windows 非法字符，别让上传把文件写到别的目录去。"""
    s = "".join("_" if (c in _BAD_NAME_CHARS or ord(c) < 32) else c for c in str(stem or ""))
    s = s.strip(" .")
    return (s or "video")[:80]


def _unique_name(target, stem, ext):
    """重名就加序号，绝不覆盖已有素材。"""
    name = stem + ext
    n = 1
    while os.path.exists(os.path.join(target, name)):
        name = "%s (%d)%s" % (stem, n, ext)
        n += 1
    return name


async def video_upload(request):
    """把视频传到当前浏览的目录。

    只允许落在 视频库 / input / output 之内——和浏览、父级导航同一套边界，
    免得从弹窗里把文件写到 ComfyUI 根目录或系统目录去。
    """
    target = _resolve_dir(request.query.get("dir", ""))
    if target is None:
        return web.json_response({"error": "目录不在 ComfyUI 根目录内"}, status=403)
    if not os.path.isdir(target):
        return web.json_response({"error": "目录不存在：%s" % target}, status=404)
    if not _under_roots(target):
        return web.json_response(
            {"error": "只能传到 视频库 / input / output 里面，当前目录不在范围内"}, status=403)

    saved = None
    try:
        reader = await request.multipart()
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name not in ("file", "image"):
                continue
            raw = part.filename or ""
            ext = os.path.splitext(raw)[1].lower()
            if ext not in VIDEO_EXTS:
                return web.json_response(
                    {"error": "不是支持的视频格式：%s" % (ext or "无扩展名")}, status=400)
            stem = _safe_name(os.path.splitext(os.path.basename(raw))[0])
            name = _unique_name(target, stem, ext)
            saved = os.path.join(target, name)
            with open(saved, "wb") as fh:
                while True:
                    chunk = await part.read_chunk(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            break
    except Exception as exc:
        if saved and os.path.isfile(saved):
            try:
                os.remove(saved)
            except Exception:
                pass
        return web.json_response({"error": "上传失败：%s" % exc}, status=500)

    if not saved or not os.path.isfile(saved) or os.path.getsize(saved) == 0:
        if saved and os.path.isfile(saved):
            try:
                os.remove(saved)
            except Exception:
                pass
        return web.json_response({"error": "没收到文件，或者文件是空的"}, status=400)

    base = _base_dir()
    index = next((v["index"] for v in _list_videos(target)
                  if os.path.normcase(v["path"]) == os.path.normcase(saved)), -1)
    return web.json_response({
        "name": os.path.basename(saved),
        "path": saved,
        "rel": _rel_path(saved, base),
        "dir": target,
        "index": index,
        "size": os.path.getsize(saved),
    })


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
    routes.post("/fxai/video/picker/upload")(video_upload)
    return True


try:
    register()
except Exception as _exc:                            # 不阻断插件加载
    print(f"[凤希AI] 视频选片路由注册失败：{_exc}")
