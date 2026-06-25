import torch

class FxAiCharacterImageByIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片列表": ("IMAGE", {"multiple": True}),
                "索引": ("INT", {
                    "default": 0,
                    "min": 0
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出图片",)
    FUNCTION = "get_index_image"
    CATEGORY = "凤希AI/工具"

    def get_index_image(self, 图片列表, 索引):
        if not 图片列表:
            return (None,)
        
        total_num = len(图片列表)
        if 索引 >= total_num:
            raise RuntimeError(f"索引超出范围，列表共{total_num}张图片，最大可用索引{total_num - 1}")
        
        return (图片列表[索引],)