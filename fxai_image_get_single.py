import torch

class FxAiImageGetSingle:
    """
    凤希AI 单张图片提取
    兼容：原生批处理张量 / 自定义列表格式
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片列表": ("IMAGE",),
                "遮罩列表": ("MASK",),
                "选取索引": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("单张图片", "单张遮罩", "宽度", "高度")
    FUNCTION = "get_single"
    CATEGORY = "凤希AI/图片"

    def get_single(self, 图片列表, 遮罩列表, 选取索引):
        # ===================== 【修复】解开嵌套 tuple =====================
        if isinstance(图片列表, tuple) and len(图片列表) == 1:
            图片列表 = 图片列表[0]
        if isinstance(遮罩列表, tuple) and len(遮罩列表) == 1:
            遮罩列表 = 遮罩列表[0]

        # ===================== 处理 图片 =====================
        if isinstance(图片列表, (list, tuple)):
            total = len(图片列表)
            idx = 选取索引 % total
            selected_img = 图片列表[idx]
        else:
            total = 图片列表.shape[0]
            idx = 选取索引 % total
            selected_img = 图片列表[idx:idx+1]

        # ===================== 处理 遮罩 =====================
        if isinstance(遮罩列表, (list, tuple)):
            total = len(遮罩列表)
            idx = 选取索引 % total
            selected_mask = 遮罩列表[idx]
        else:
            total = 遮罩列表.shape[0]
            idx = 选取索引 % total
            selected_mask = 遮罩列表[idx:idx+1]

        # ===================== 获取尺寸 =====================
        height = selected_img.shape[1]
        width = selected_img.shape[2]

        return (selected_img, selected_mask, width, height)