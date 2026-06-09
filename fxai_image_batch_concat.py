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

    def to_batch(self, img):
        if img is None:
            return None
        if isinstance(img, list):
            return img
        return [img]

    def concat_batch(self, 图片列表A, 图片列表B):
        list_a = self.to_batch(图片列表A)
        list_b = self.to_batch(图片列表B)
        
        combined_list = []
        if list_a is not None:
            combined_list.extend(list_a)
        if list_b is not None:
            combined_list.extend(list_b)

        total = len(combined_list)
        return (combined_list, total)