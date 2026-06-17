import os
import math
import torch
import numpy as np
from PIL import Image

class ImageSizeController:
    """图片宽高控制类 - 统一管理图片尺寸、缩放、拼接逻辑"""
    # 默认画布尺寸（可通过初始化覆盖）
    DEFAULT_CANVAS_W = 1024
    DEFAULT_CANVAS_H = 1024
    # 支持的图片格式
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

    def __init__(self, canvas_w=None, canvas_h=None, bg_color=None):
        """
        初始化图片尺寸控制器
        :param canvas_w: 目标宽度（所有缩放/拉伸都会用这个尺寸）
        :param canvas_h: 目标高度
        :param bg_color: 画布背景色
                    - None / 不传：默认纯透明（RGBA 0,0,0,0）
                    - 黑色：(0,0,0)
                    - 白色：(255,255,255)
                    - 带透明：(0,0,0,0) / (255,255,255,128)
        """
        self.canvas_w = canvas_w or self.DEFAULT_CANVAS_W
        self.canvas_h = canvas_h or self.DEFAULT_CANVAS_H
        self.bg_color = bg_color  # 背景颜色，默认透明

    def load_single_image(self, image_path):
        """加载单张图片为张量（仅加载，不缩放）"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片路径不存在: {image_path}")
        if not image_path.lower().endswith(self.IMAGE_EXTENSIONS):
            raise ValueError(f"不支持的图片格式，仅支持: {self.IMAGE_EXTENSIONS}")
        
        # 统一加载为 RGBA 支持透明通道
        img = Image.open(image_path).convert("RGBA")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np)[None,]  # [1, H, W, 4]
        return img_tensor

    def fit_to_canvas(self, tensor_img):
        """
        等比例缩放到实例设定的画布尺寸，居中 + 自定义背景，不变形
        自动用 self.canvas_w / self.canvas_h / self.bg_color
        """
        target_w = self.canvas_w
        target_h = self.canvas_h
        
        _, src_h, src_w, src_c = tensor_img.shape
        if src_w == target_w and src_h == target_h:
            return tensor_img

        # 张量转PIL
        img = tensor_img.squeeze(0).cpu().numpy()
        pil_img = Image.fromarray((img * 255).astype(np.uint8))

        # 等比缩放：取小的，完整显示
        scale = min(target_w / src_w, target_h / src_h)
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)
        resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 背景颜色/透明
        if self.bg_color is None:
            canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        else:
            canvas = Image.new("RGBA", (target_w, target_h), self.bg_color)
        
        # 居中粘贴
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y), mask=resized)

        # 转回张量
        out_np = np.array(canvas).astype(np.float32) / 255.0
        return torch.from_numpy(out_np)[None,]

    def crop_fill_to_canvas(self, tensor_img):
        """
        【按你要求：目标宽/原图宽，目标高/原图高，取大缩放比 → 铺满一边，另一边填充】
        1. 计算 target_w / src_w
        2. 计算 target_h / src_h
        3. 取 更大的缩放比例 缩放图片
        4. 铺满画布一条边，另一条不足 → 居中填充背景色
        5. 绝不裁剪图片，不变形
        """
        target_w = self.canvas_w
        target_h = self.canvas_h
        
        _, src_h, src_w, _ = tensor_img.shape
        if src_w == target_w and src_h == target_h:
            return tensor_img

        # 张量 → PIL
        img = tensor_img.squeeze(0).cpu().numpy()
        pil_img = Image.fromarray((img * 255).astype(np.uint8))

        # ===================== 你说的正确算法 =====================
        scale_w = target_w / src_w   # 目标宽 / 原图宽
        scale_h = target_h / src_h   # 目标高 / 原图高
        scale = max(scale_w, scale_h)# 取大的那个缩放！

        # 等比缩放
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)
        resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 创建画布
        if self.bg_color is None:
            canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        else:
            canvas = Image.new("RGBA", (target_w, target_h), self.bg_color)

        # 居中，不足部分自动填充
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y), mask=resized)

        # 转回张量
        out_np = np.array(canvas).astype(np.float32) / 255.0
        return torch.from_numpy(out_np)[None,]

    def stretch_to_size(self, tensor_img):
        """
        强制拉伸到实例设定的尺寸
        自动用 self.canvas_w / self.canvas_h
        """
        target_w = self.canvas_w
        target_h = self.canvas_h

        _, src_h, src_w, _ = tensor_img.shape
        if src_w == target_w and src_h == target_h:
            return tensor_img

        # 张量 → PIL
        img = tensor_img.squeeze(0).cpu().numpy()
        pil_img = Image.fromarray((img * 255).astype(np.uint8))

        # 强制拉伸
        resized_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # 转回张量
        out_np = np.array(resized_img).astype(np.float32) / 255.0
        return torch.from_numpy(out_np)[None,]

    def scale_down_by_factor(self, tensor_img, scale_factor):
        """
        按指定倍数等比例缩小图片（只缩小，不放大，保证不变形）
        :param tensor_img: 输入图片张量 [1, H, W, 4]
        :param scale_factor: 缩小倍数（必须 ≥ 1，例如 1=不缩放，2=缩小2倍，4=缩小4倍）
        :return: 缩小后的图片张量
        """
        if scale_factor == 1.0:
            return tensor_img
        if not isinstance(scale_factor, (int, float)) or scale_factor < 1.0:
            raise ValueError("缩小倍数必须是大于等于 1 的数字，例如 1、2、2.5、4")

        _, src_h, src_w, _ = tensor_img.shape
        
        new_w = int(src_w / scale_factor)
        new_h = int(src_h / scale_factor)
        
        new_w = max(1, new_w)
        new_h = max(1, new_h)

        img = tensor_img.squeeze(0).cpu().numpy()
        pil_img = Image.fromarray((img * 255).astype(np.uint8))

        resized_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        out_np = np.array(resized_img).astype(np.float32) / 255.0
        return torch.from_numpy(out_np)[None,]

    def grid_concat_images(self, images_list, cols_mode="auto"):
        """网格拼接图片列表（自动统一尺寸 + 补全网格空白）"""
        n = len(images_list)
        if n == 0:
            raise ValueError("图片列表不能为空")

        processed = [self.crop_fill_to_canvas(img) for img in images_list]

        if cols_mode == "fixed_2":
            cols = 2
        else:
            cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        blank = torch.zeros_like(processed[0])
        while len(processed) < rows * cols:
            processed.append(blank)

        # 第四步：拼接
        rows_tensor = []
        for i in range(rows):
            row_imgs = processed[i*cols : (i+1)*cols]
            rows_tensor.append(torch.cat(row_imgs, dim=2))

        return torch.cat(rows_tensor, dim=1)

    def get_image_width_height(self,img_tensor):
        h = img_tensor.shape[1]
        w = img_tensor.shape[2]
        return h, w

# ===================== 全局导出（方法名完全不变） =====================
_global_size_controller = ImageSizeController(bg_color=None)
load_single_image = _global_size_controller.load_single_image
get_image_width_height = _global_size_controller.get_image_width_height
fit_to_canvas = _global_size_controller.fit_to_canvas
crop_fill_to_canvas = _global_size_controller.crop_fill_to_canvas
stretch_to_size = _global_size_controller.stretch_to_size
scale_down_by_factor = _global_size_controller.scale_down_by_factor
grid_concat_images = _global_size_controller.grid_concat_images
IMAGE_EXTENSIONS = ImageSizeController.IMAGE_EXTENSIONS