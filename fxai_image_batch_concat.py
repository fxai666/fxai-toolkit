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
    RETURN_NAMES = ("合并图片", "总数量")
    FUNCTION = "concat_batch"
    CATEGORY = "凤希AI/图片"

    # 直接照搬它这个万能转换方法！
    def to_batch(self, img):
        if img is None:
            return None
        if isinstance(img, list):
            return torch.cat(img, dim=0) if img else None
        return img

    def concat_batch(self, 图片列表A, 图片列表B):
        # 统一转成标准张量
        a_tensor = self.to_batch(图片列表A)
        b_tensor = self.to_batch(图片列表B)

        # 收集非空张量
        tensor_list = []
        if a_tensor is not None:
            tensor_list.append(a_tensor)
        if b_tensor is not None:
            tensor_list.append(b_tensor)

        # 最终合并
        combined = torch.cat(tensor_list, dim=0)
        total = combined.shape[0]
        return (combined, total)