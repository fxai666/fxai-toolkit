import torch

class FxAiImageBatchConcat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片列表A": ("IMAGE",),
                "图片列表B": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("图片列表", "总数量")
    FUNCTION = "concat_batch"
    CATEGORY = "凤希AI/图片"

    def concat_batch(self, 图片列表A, 图片列表B):
        # 空列表
        batch_list = []

        # 按顺序追加：A 在前，B 在后
        if 图片列表A is not None:
            batch_list.append(图片列表A)
        
        if 图片列表B is not None:
            batch_list.append(图片列表B)

        # 空值判断
        if not batch_list:
            raise RuntimeError("图片列表A和图片列表B均为空")

        # 合并成一个张量
        combined = torch.cat(batch_list, dim=0)
        total_count = combined.shape[0]

        return (combined, total_count)