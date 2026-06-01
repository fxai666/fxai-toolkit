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
                "": ("*", {}),
                "是否卸载模型": ("BOOLEAN", {"default": True, "label_on": "是", "label_off": "否"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "release_all_resources"
    CATEGORY = "凤希AI / 工具"
    OUTPUT_NODE = True

    def release_all_resources(self, _, 是否卸载模型):
        try:
            comfy.model_management.soft_empty_cache()

            if 是否卸载模型:
                comfy.model_management.unload_all_models()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

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
            print("[凤希AI] 所有资源已完全释放 ✅")
        except Exception:
            pass