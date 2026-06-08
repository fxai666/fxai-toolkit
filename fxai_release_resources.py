import os
import gc
import torch
import psutil
import subprocess
import comfy.model_management

class FxAiReleaseResources:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "输入": ("*", {}),  # 万能输入，可接任何节点
            },
            "optional": {
                "卸载所有模型": ("BOOLEAN", {"default": True, "label_on": "是", "label_off": "否"}),
                "查杀子进程": ("BOOLEAN", {"default": True, "label_on": "是", "label_off": "否"}),
                "清空CUDA缓存": ("BOOLEAN", {"default": True, "label_on": "是", "label_off": "否"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "release_all_resources"
    CATEGORY = "凤希AI/工具"
    OUTPUT_NODE = True

    def release_all_resources(self, 输入, 卸载所有模型=True, 查杀子进程=True, 清空CUDA缓存=True):
        try:
            print("[凤希AI] 开始执行完整资源释放...")

            # 1. ComfyUI 模型管理清理
            comfy.model_management.soft_empty_cache()

            # 2. 卸载所有模型
            if 卸载所有模型:
                comfy.model_management.unload_all_models()
                comfy.model_management.free_memory(0, torch.device("cuda"))

            # 3. CUDA 显卡资源释放
            if 清空CUDA缓存 and torch.cuda.is_available():
                with torch.no_grad():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

            # 4. 查杀所有子进程（FFmpeg / 外部进程）
            if 查杀子进程:
                try:
                    current_pid = os.getpid()
                    process = psutil.Process(current_pid)
                    children = process.children(recursive=True)

                    for child in children:
                        try:
                            child.kill()
                            child.wait(timeout=2)
                        except Exception:
                            pass

                    # 额外清理僵尸/残留进程
                    for proc in psutil.process_iter(['pid', 'name']):
                        try:
                            if 'ffmpeg' in proc.info['name'].lower():
                                proc.kill()
                        except Exception:
                            pass

                except Exception as e:
                    print(f"[凤希AI] 子进程清理异常：{str(e)}")

            # 5. Python 内存垃圾回收
            gc.collect()

            print("[凤希AI] 资源释放完成 ✅")

        except Exception as e:
            print(f"[凤希AI] 资源释放失败：{str(e)}")

        return ()