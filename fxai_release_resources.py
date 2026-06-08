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
            print("[凤希AI] 开始回收主进程内存与虚拟提交空间")
            # 1. ComfyUI内置缓存清理
            comfy.model_management.soft_empty_cache()

            # 2. 卸载模型，释放大块虚拟地址
            if 卸载所有模型:
                comfy.model_management.unload_all_models()

            # 3. 同步并清空CUDA映射虚拟内存
            if 清空CUDA显存 and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            # 4. 强制完整垃圾回收，销毁无用对象
            gc.collect()

            # Windows专属：双重操作，同时降物理内存+虚拟提交大小
            if os.name == "nt":
                try:
                    import ctypes
                    from ctypes import wintypes
                    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                    hProc = kernel32.GetCurrentProcess()

                    # 结构体用于遍历本进程全部虚拟内存区域
                    class MEMORY_BASIC_INFORMATION(ctypes.Structure):
                        _fields_ = [
                            ("BaseAddress", wintypes.LPVOID),
                            ("AllocationBase", wintypes.LPVOID),
                            ("AllocationProtect", wintypes.DWORD),
                            ("RegionSize", wintypes.SIZE_T),
                            ("State", wintypes.DWORD),
                            ("Protect", wintypes.DWORD),
                            ("Type", wintypes.DWORD),
                        ]
                    mbi = MEMORY_BASIC_INFORMATION()
                    addr = 0
                    MEM_COMMIT = 0x1000
                    MEM_DECOMMIT = 0x4000
                    PAGE_READWRITE = 0x04
                    MEM_PRIVATE = 0x20000

                    # 遍历私有读写已提交堆，取消闲置虚拟内存提交（直接降低提交大小）
                    while kernel32.VirtualQueryEx(hProc, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)):
                        # 只处理私有、读写、已提交的堆内存，不碰代码、dll、CUDA映射区
                        if mbi.State == MEM_COMMIT and mbi.Protect == PAGE_READWRITE and mbi.Type == MEM_PRIVATE:
                            kernel32.VirtualFree(mbi.BaseAddress, mbi.RegionSize, MEM_DECOMMIT)
                        addr += mbi.RegionSize

                    # 二次压缩物理工作集
                    kernel32.SetProcessWorkingSetSize(hProc, -1, -1)
                except Exception as winErr:
                    print(f"[凤希AI Windows内存优化提示] {str(winErr)}")

            print("[凤希AI] ✅ 资源释放执行完毕")
        except Exception as e:
            print(f"[凤希AI] 释放异常：{str(e)}")
        return ()