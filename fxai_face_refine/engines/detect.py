# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License

"""
凤希AI 人脸检测 / 跟踪
=====================

检测器本身用 insightface 的 buffalo_l（det_10g 检测 + w600k_r50 识别 +
genderage 性别年龄），这是第三方预训练模型，不属于 VOSR2 那套，没必要重造。
这里重写的是**怎么用它**：

问题
----
逐帧对整幅 1280×768 跑 640×640 检测，远景里只有 43px 的小脸 ——
等于让检测器在一张被缩掉的图上找一个极小的目标，既慢又容易掉。

做法
----
**ROI 跟踪**：人脸在相邻帧之间的位移远小于画幅。拿到上一帧的框后，
只把框周围一片（默认放大到 2.2 倍）裁出来送去检测 —— 脸在 ROI 里的
相对面积大得多，检测又快又稳。ROI 内检不到（切镜、大幅运动）才回退全图。

**整片预扫**：既然要处理整段视频，先扫一遍把每帧的框和关键点缓存下来，
比"处理到哪帧才检测哪帧"省掉一次重复的 IO 遍历，也方便提前判断
哪些帧根本没脸、可以直接跳过。
"""

import os
import threading

import numpy as np

_LOCK = threading.Lock()
_CACHE: dict = {}


def _ensure_cuda_dlls():
    if _CACHE.get("cuda_dlls") is not None:
        return _CACHE["cuda_dlls"]
    ok = False
    try:
        import torch
        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(lib):
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(lib)
            os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
            ok = True
    except Exception:
        ok = False
    _CACHE["cuda_dlls"] = ok
    return ok


class Face:
    """一帧里的一张脸：框、五点、身份向量、性别。只留后面真正用得到的字段。"""
    __slots__ = ("bbox", "kps", "score", "embedding", "sex")

    def __init__(self, bbox, kps, score, embedding, sex=None):
        self.bbox = bbox            # np.float32[4] x1,y1,x2,y2
        self.kps = kps              # np.float32[5,2]
        self.score = float(score)
        self.embedding = embedding  # 已归一化，可直接点积算余弦
        self.sex = sex              # 'M' / 'F'，没跑性别模型时为 None

    @property
    def w(self):
        return float(self.bbox[2] - self.bbox[0])

    @property
    def h(self):
        return float(self.bbox[3] - self.bbox[1])

    @property
    def area(self):
        return self.w * self.h

    @property
    def center(self):
        return ((self.bbox[0] + self.bbox[2]) * 0.5, (self.bbox[1] + self.bbox[3]) * 0.5)


class Detector:
    def __init__(self, device_id=0, det_size=640, verbose=True):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "缺 insightface。在 ComfyUI 的 Python 环境里执行：\n"
                "    pip install insightface onnxruntime-gpu\n"
                "并把 buffalo_l 放到 models/insightface/buffalo_l/"
            ) from exc

        from . import paths
        bdir = paths.buffalo_dir()
        if not os.path.isdir(bdir):
            raise RuntimeError(
                f"找不到 buffalo_l 检测模型：\n    {bdir}\n"
                f"需要 det_10g.onnx / w600k_r50.onnx 等 5 个文件。\n"
                f"（放在这里就不会触发 insightface 的在线下载）"
            )

        _ensure_cuda_dlls()
        import onnxruntime as ort
        provs = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            provs = [("CUDAExecutionProvider", {"device_id": int(device_id)}),
                     "CPUExecutionProvider"]

        app = None
        if provs[0] != "CPUExecutionProvider":
            try:
                cand = FaceAnalysis(name=bdir,
                                    allowed_modules=["detection", "recognition", "genderage"],
                                    providers=provs)
                cand.prepare(ctx_id=int(device_id), det_size=(det_size, det_size))
                got = cand.det_model.session.get_providers() if cand.det_model else []
                if "CUDAExecutionProvider" in got:
                    app = cand
                else:
                    raise RuntimeError(f"实际 provider = {got}")
            except Exception as exc:
                print(f"⚠️[凤希AI] 人脸检测 GPU 不可用，退回 CPU: {exc}")
        if app is None:
            app = FaceAnalysis(name=bdir,
                               allowed_modules=["detection", "recognition", "genderage"],
                               providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(det_size, det_size))

        self.app = app
        self.det_size = det_size
        self.device = "CUDA" if "CUDAExecutionProvider" in (
            app.det_model.session.get_providers() if app.det_model else []) else "CPU"
        if verbose:
            print(f"[凤希AI·检测] buffalo_l 就绪  {self.device}  det_size={det_size}")

    # ---- 原始检测 ----
    def raw(self, img):
        """img: BGR uint8 或 RGB float[0,1]；返回 insightface 原生 Face 列表"""
        if img is None or getattr(img, "size", 0) == 0:
            return []
        return self.app.get(img) or []

    def detect(self, img, order=0):
        """按面积从大到小取第 order 张脸，包装成我们的 Face"""
        faces = self.raw(img)
        if not faces:
            return None
        faces.sort(key=lambda f: -((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
        if order >= len(faces):
            return None
        return self._wrap(faces[order])

    def detect_all(self, img):
        return [self._wrap(f) for f in self.raw(img)]

    @staticmethod
    def _wrap(f):
        emb = getattr(f, "normed_embedding", None)
        return Face(np.asarray(f.bbox, np.float32),
                    np.asarray(f.kps, np.float32),
                    float(getattr(f, "det_score", 0.0)),
                    None if emb is None else np.asarray(emb, np.float32),
                    getattr(f, "sex", None))

    # ---- ROI 跟踪 ----
    def track(self, img, prev, margin=2.2, order=0):
        """用上一帧的框做 ROI 检测。检不到就回退全图。

        prev: 上一帧的 Face（或 None → 直接全图）
        margin: ROI 相对脸宽的扩展倍数，2.2 大约能容忍每秒 20% 画幅的位移
        """
        h, w = img.shape[:2]
        if prev is None:
            return self.detect(img, order)

        cx, cy = prev.center
        half = max(prev.w, prev.h) * margin * 0.5
        x1 = int(max(0, cx - half))
        y1 = int(max(0, cy - half))
        x2 = int(min(w, cx + half))
        y2 = int(min(h, cy + half))
        # ROI 太小就别折腾了，直接整图
        if x2 - x1 < 32 or y2 - y1 < 32:
            return self.detect(img, order)

        roi = img[y1:y2, x1:x2]
        faces = self.raw(roi)
        if not faces:
            return self.detect(img, order)

        faces.sort(key=lambda f: -((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
        if order >= len(faces):
            return self.detect(img, order)
        best = faces[order]

        # 把 ROI 坐标平移回全图
        box = best.bbox.copy()
        box[0] += x1; box[2] += x1
        box[1] += y1; box[3] += y1
        kps = best.kps.copy()
        kps[:, 0] += x1
        kps[:, 1] += y1
        emb = getattr(best, "normed_embedding", None)
        return Face(box.astype(np.float32), kps.astype(np.float32),
                    float(getattr(best, "det_score", 0.0)),
                    None if emb is None else np.asarray(emb, np.float32),
                    getattr(best, "sex", None))

    # ---- 按身份锁定 ----
    def track_by_identity(self, img, prev, target_emb, threshold=0.40,
                          margin=2.2, min_size=16, fallback=False):
        """在画面里找出「指定身份」的那张脸，而不是面积最大的那张。

        为什么需要它
        ------------
        按面积挑脸（detect 的 order）在多人画面里会换错人：谁走近镜头就换谁。
        43px 级的小脸放大缩小来回折腾，身份向量与本人的相似度仍明显高于与
        其他人的相似度 —— 远景脸虽然糊，但"是谁"这件事可辨。

        所以这里改成比身份向量：目标向量全片固定（开场锁定或参考图提取），
        不管谁站到前面，都只认那一个人。

        返回 (Face 或 None, 本帧最相似度)。相似度低于 threshold 时返回 None：
        宁可不换，也不把配角换成主角的脸。

        fallback=True：**不做阈值拒绝**，永远返回本帧最像的那张脸，相似度原样报出。
        给「认不出是谁也照样要修」的场景用 —— 远景小脸（35~43px）提不出稳定身份
        向量，相似度只有 0.01~0.20，按阈值过滤等于整片不修。放宽确认线之后，
        多人画面里仍然修「最像主角的那张」，比按面积挑准得多。
        调用方自己拿返回的 sim 判断过没过确认线即可，一遍扫描两套结果。
        """
        if target_emb is None:
            return self.detect(img, order=0), 0.0
        target = np.asarray(target_emb, np.float32)

        def scan(x1, y1, x2, y2):
            sub = img[y1:y2, x1:x2]
            if sub.size == 0:
                return []
            out = []
            for f in self.raw(sub):
                emb = getattr(f, "normed_embedding", None)
                if emb is None:
                    continue
                box = np.asarray(f.bbox, np.float32).copy()
                if box[2] - box[0] < min_size:
                    continue
                box[0] += x1; box[2] += x1
                box[1] += y1; box[3] += y1
                kps = np.asarray(f.kps, np.float32).copy()
                kps[:, 0] += x1; kps[:, 1] += y1
                e = np.asarray(emb, np.float32)
                out.append((float(np.dot(e, target)),
                            Face(box, kps, float(getattr(f, "det_score", 0.0)), e,
                                 getattr(f, "sex", None))))
            return out

        h, w = img.shape[:2]
        cands = []
        if prev is not None:
            cx, cy = prev.center
            half = max(prev.w, prev.h) * margin * 0.5
            x1 = int(max(0, cx - half)); y1 = int(max(0, cy - half))
            x2 = int(min(w, cx + half)); y2 = int(min(h, cy + half))
            if x2 - x1 >= 32 and y2 - y1 >= 32:
                cands = scan(x1, y1, x2, y2)
                # ROI 里最像的都不够像 -> 目标可能移出 ROI 或切镜了，回退全图。
                # 这个判断**和 fallback 无关**：fallback 只是不拒绝结果，
                # 不代表 ROI 可信。去掉它会让 prev 锁死在错误的人身上。
                if not cands or max(c[0] for c in cands) < threshold:
                    cands = scan(0, 0, w, h)
        else:
            cands = scan(0, 0, w, h)

        if not cands:
            return None, -1.0
        best_sim, best = max(cands, key=lambda c: c[0])
        if not fallback and best_sim < threshold:
            return None, best_sim
        return best, best_sim

    # ---- 整片预扫 ----
    def scan(self, frames_iter, order=0, margin=2.2, progress=None):
        """对一段帧流做 ROI 跟踪，返回 [(face_or_None)] 与原帧一一对应。

        frames_iter 只要求可迭代（节点里可以直接喂 torch 张量的切片），
        这样不必先把整片读进内存。
        """
        out = []
        prev = None
        for i, img in enumerate(frames_iter):
            f = self.track(img, prev, margin=margin, order=order)
            if f is not None:
                prev = f
            out.append(f)
            if progress is not None:
                progress(i, f)
        return out


def get_detector(device_id=0, det_size=640, verbose=True):
    key = (int(device_id), int(det_size))
    with _LOCK:
        d = _CACHE.get(key)
        if d is None:
            d = Detector(device_id=device_id, det_size=det_size, verbose=verbose)
            _CACHE[key] = d
        return d


def cosine(a, b):
    if a is None or b is None:
        return -1.0
    return float(np.dot(a, b))
