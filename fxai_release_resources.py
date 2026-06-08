import os
import gc
import torch
import comfy.model_management

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
            # 1. ComfyUI 缓存清理
            comfy.model_management.soft_empty_cache()

            # 2. 卸载模型
            if 卸载所有模型:
                comfy.model_management.unload_all_models()

            # 3. 清空 CUDA 显存
            if 清空CUDA显存 and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            # 4. Python 内存垃圾回收
            gc.collect()

            # --------------------------
            # Windows 虚拟内存释放（修复版）
            # --------------------------
            if os.name == "nt":
                try:
                    import ctypes
                    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                    handle = kernel32.GetCurrentProcess()
                    kernel32.SetProcessWorkingSetSize(handle, -1, -1)
                except Exception:
                    pass

            print("[凤希AI] ✅ 资源释放完成！")

        except Exception as e:
            print(f"[凤希AI] 释放出错：{str(e)}")

        return ()