import os
import gc
import torch
import comfy.model_management


def release_resources(unload_models=True, empty_cuda=True, shrink_working_set=True):
    """释放显存 / 内存。全包唯一的实现，各个清理节点都调它。

    :param unload_models: 卸载 ComfyUI 里所有已加载的模型
    :param empty_cuda: 清 CUDA 显存池 + IPC 句柄
    :param shrink_working_set: Windows 上把进程工作集还给系统
    """
    # 1. ComfyUI 缓存清理
    comfy.model_management.soft_empty_cache()

    # 2. 卸载模型
    if unload_models:
        comfy.model_management.unload_all_models()

    # 3. 清空 CUDA 显存
    if empty_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # 4. Python 内存垃圾回收
    gc.collect()

    # 5. Windows 虚拟内存释放
    if shrink_working_set and os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.GetCurrentProcess()
            kernel32.SetProcessWorkingSetSize(handle, -1, -1)
        except Exception:
            pass


class FxAiReleaseResources:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "输入": ("*", {}),
            },
            "optional": {
                "卸载所有模型": ("BOOLEAN", {"default": True, "label_on": "是", "label_off": "否"}),
                "清空CUDA显存": ("BOOLEAN", {"default": True, "label_on": "是", "label_off": "否"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "release_all_resources"
    CATEGORY = "凤希AI/工具"
    OUTPUT_NODE = True

    def release_all_resources(self, 输入, 卸载所有模型=True, 清空CUDA显存=True):
        try:
            release_resources(unload_models=卸载所有模型, empty_cuda=清空CUDA显存)
            print("[凤希AI] ✅ 资源释放完成！")
        except Exception as e:
            print(f"[凤希AI] 释放出错：{str(e)}")

        return ()
