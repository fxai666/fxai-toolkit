# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License

"""
凤希AI 自研换脸引擎
===================

只用 `inswapper_128.onnx` 这一个模型权重，前处理、对齐、推理、融合全部自己实现，
不调用 `insightface.model_zoo.INSwapper` 的封装。

对外接口
--------
只做 `paste_back=False` 那部分：输出一张 128×128 的换脸结果 + 对齐矩阵，
合成由调用方一次完成（下游用椭圆 mask 贴回，方形 mask 的合成产物会被丢弃）。

身份投影矩阵（emap）从 ONNX 权重里读出后缓存成 .npy，后续加载直接复用，
不重复解析整个 protobuf。

物理限制
--------
inswapper 是 128×128 的 GAN，输出只有 128 空间的信息量。贴回 512 的超分结果时，
那块脸的纹理被限制在 128 空间 —— 超分救回来的细节会被换脸冲掉一部分。

因此身份与纹理分开处理（`texture_keep` 参数）：
    final = 换脸结果 + (超分结果 - 低频(超分结果)) × texture_keep
身份由 inswapper 决定，纹理由 VOSR2 决定，两者互不覆盖。texture_keep=0 为不注入。
"""

import os
import threading

import numpy as np

_LOCK = threading.Lock()
_CACHE: dict = {}

# 对齐模板：inswapper 官方在 112 模板上按比例放大，FaceFusion 用的是
# 专门给 128 调过的五点位置（脸更大、额头更靠上）。同一帧同一身份的判据：
#   112 模板缩放 -> 相似度 0.863
#   arcface_128  -> 相似度 0.914，且五官更立体
# 所以这里直接用 FaceFusion 的归一化模板，使用时乘以目标边长。
_TEMPLATE_128 = np.array([
    [0.36167656, 0.40387734],
    [0.63696719, 0.40235469],
    [0.50019687, 0.56044219],
    [0.38710391, 0.72160547],
    [0.61507734, 0.72034453],
], dtype=np.float32)


def _ensure_cuda_dlls():
    """onnxruntime-gpu 的 CUDA EP 要求 cuDNN 9 + CUDA 13，正好与 torch 2.10+cu130
    自带的那套一致，但 ORT 不会自己去找 torch/lib，必须先挂进 DLL 搜索路径，
    否则加载 providers_cuda.dll 时报 cudnn64_9.dll 缺失，然后静默退回 CPU。"""
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


def _emap(model_path):
    """取身份投影矩阵。解析一次后缓存成 npy —— 省掉每次构造都重解析 529MB ONNX。"""
    cache = os.path.join(os.path.dirname(model_path), "inswapper_emap.npy")
    if os.path.isfile(cache):
        try:
            return np.load(cache)
        except Exception:
            pass
    import onnx
    from onnx import numpy_helper
    m = onnx.load(model_path)
    arr = numpy_helper.to_array(m.graph.initializer[-1]).astype(np.float32)
    try:
        np.save(cache, arr)
    except Exception:
        pass
    return arr


def _umeyama_like(src, dst):
    """5 点相似变换（旋转 + 等比缩放 + 平移），返回 2×3 仿射矩阵。

    语义与 insightface 内部用的 skimage `SimilarityTransform.estimate(src, dst)`
    一致：求「把 src 映射到 dst」的变换。
    注意 cv2 的 `estimateAffinePartial2D(from_, to)` 也是这个方向，
    所以这里传参顺序必须是 (人脸关键点, 模板) —— 反了会得到一堆错位的脸。
    """
    import cv2
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
    if M is None:
        raise RuntimeError("人脸关键点求解对齐矩阵失败")
    return M.astype(np.float32)


class SwapEngine:
    """inswapper_128 的轻封装：一次建 session，反复换脸。"""

    def __init__(self, model_path, device_id=0, verbose=True):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"找不到换脸模型：{model_path}")
        self.model_path = model_path
        self.size = 128

        _ensure_cuda_dlls()
        import onnxruntime as ort
        provs = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            provs = [("CUDAExecutionProvider", {"device_id": int(device_id)}),
                     "CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        self.session = ort.InferenceSession(model_path, sess_options=so, providers=provs)
        self.in_names = [i.name for i in self.session.get_inputs()]
        self.out_names = [o.name for o in self.session.get_outputs()]
        self.providers = self.session.get_providers()
        self.emap = _emap(model_path)
        self._src_latent = None
        self._src_emb = None
        if verbose:
            dev = "CUDA" if "CUDAExecutionProvider" in self.providers else "CPU"
            print(f"[凤希AI·换脸] inswapper_128 就绪  {dev}  emap{self.emap.shape}")

    # ---- 身份 ----
    def encode_source(self, normed_embedding):
        """把身份向量预计算成 512 维 latent（走 emap 映射 + 归一化）。

        多角色场景一次算好所有人的 latent 缓存着，每次换脸按需传入，
        就不用反复切换引擎的全局身份状态，也就不会串人。
        """
        emb = np.asarray(normed_embedding, dtype=np.float32).reshape(1, -1)
        lat = np.dot(emb, self.emap)
        lat /= np.linalg.norm(lat)
        return lat.astype(np.float32)

    def set_source(self, normed_embedding):
        """锁定参考身份。全片只调一次，逐帧更新身份向量会出现闪烁。"""
        self._src_emb = np.asarray(normed_embedding, dtype=np.float32).reshape(1, -1)
        self._src_latent = self.encode_source(normed_embedding)

    @property
    def has_source(self):
        return self._src_latent is not None

    # ---- 换脸 ----
    def algorithm(self, img_bgr, kps, src_latent=None):
        """把脸对齐到 128 空间并推理，返回 (128×128 换脸结果 BGR, 2×3 对齐矩阵)。

        对齐矩阵 M 把「原图坐标」映射到「128 空间坐标」，
        调用方用 `cv2.invertAffineTransform(M)` 就能把结果贴回原图。

        src_latent: 指定这次用哪个身份（多人各换各的）。不给就用 set_source 设的那个。
        """
        latent = self._src_latent if src_latent is None else np.asarray(src_latent, np.float32)
        if latent is None:
            raise RuntimeError("还没设参考身份，先调 set_source()")
        import cv2

        dst = _TEMPLATE_128 * float(self.size)
        # 关键点 -> 128 模板。方向必须一致：反了身份相似度从 0.89 掉到 0.06。
        M = _umeyama_like(np.asarray(kps, dtype=np.float32).reshape(5, 2), dst)
        aligned = cv2.warpAffine(img_bgr, M, (self.size, self.size), borderValue=0.0)

        # HWC BGR uint8 -> NCHW RGB float32（等价 cv2.dnn.blobFromImage，但少一次中转）
        blob = np.ascontiguousarray(
            aligned[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
        blob = (blob - 0.0) / 255.0

        pred = self.session.run(self.out_names, {
            self.in_names[0]: blob,
            self.in_names[1]: latent,
        })[0]
        fake = pred.transpose(0, 2, 3, 1)[0]
        fake = np.clip(255.0 * fake, 0, 255).astype(np.uint8)[:, :, ::-1]
        return np.ascontiguousarray(fake), M

    def swap_face_only(self, img_bgr, kps):
        """只要那张 128 的脸，不管贴回 —— 贴回由调用方用自己的 mask 做。"""
        return self.algorithm(img_bgr, kps)


def get_swapper(model_path=None, device_id=0, verbose=True):
    if model_path is None:
        from . import paths
        model_path = paths.inswapper_path()

    key = (model_path, int(device_id))
    with _LOCK:
        eng = _CACHE.get(key)
        if eng is None:
            eng = SwapEngine(model_path, device_id=device_id, verbose=verbose)
            _CACHE[key] = eng
        return eng


def release():
    with _LOCK:
        _CACHE.clear()
