import os
import torch
from fxai_image_utils import get_image_width_height

class FxAiImageSelectorByIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片列表": ("IMAGE",),
                "索引": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("图片", "宽度", "高度")
    FUNCTION = "get_image_by_index"
    CATEGORY = "凤希AI/角色"

    def get_image_by_index(self, 图片列表, 索引):
        tensor_list = []
        if isinstance(图片列表, list):
            tensor_list = 图片列表
        elif isinstance(图片列表, torch.Tensor) and 图片列表.dim() == 4:
            for i in range(图片列表.shape[0]):
                tensor_list.append(图片列表[i:i+1])

        if len(tensor_list) == 0:
            raise RuntimeError("输入的图片列表不能为空")
        
        max_index = len(tensor_list) - 1
        if 索引 < 0 or 索引 > max_index:
            raise RuntimeError(f"索引超出范围，当前图片总数：{len(tensor_list)}，有效索引范围 0 ~ {max_index}")
        
        target_img = tensor_list[索引]
        高度, 宽度 = get_image_width_height(target_img)

        return (target_img, 宽度, 高度)