import os
import gc
import torch
import psutil
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
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "release_all_resources"
    CATEGORY = "凤希AI/工具"
    OUTPUT_NODE = True

    # 参数名必须和上面 INPUT_TYPES 完全对应
    def release_all_resources(self, 输入, 卸载所有模型=True):
        try:
            # 清空缓存
            comfy.model_management.soft_empty_cache()

            # 卸载模型
            if 卸载所有模型:
                comfy.model_management.unload_all_models()

            # CUDA 显存清理
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            # 清理僵尸进程
            try:
                current_pid = os.getpid()
                process = psutil.Process(current_pid)
                children = process.children(recursive=True)
                for child in children:
                    try:
                        child.kill()
                    except Exception:
                        pass
            except Exception:
                pass

            gc.collect()
            print("[凤希AI] 资源释放完成 ✅")

        except Exception as e:
            print(f"[凤希AI] 资源释放出错：{str(e)}")