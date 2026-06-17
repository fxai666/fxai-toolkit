class FxAiPhotoSizeConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "尺寸设置": ("BOOLEAN", {
                    "default": True,
                    "label_on": "自定义宽高",
                    "label_off": "选择预设"
                }),
                "预设": (
                    [
                        "1寸证件照 256×384",
                        "大一寸护照 384×560",
                        "2寸证件照 416×576",
                        "5寸(3R) 896×1280",
                        "6寸(4R) 1280×1792",
                        "7寸(5R) 1536×2176",
                        "8寸(6R) 1792×2432",
                        "10寸 2304×2880",
                        "正方形 1024×1024",
                        "竖屏 1080×1920",
                        "横屏 1920×1080"
                    ],
                    {"default": "竖屏 1080×1920"}
                ),
                "宽度": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "display": "number"
                }),
                "高度": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "display": "number"
                }),
                "反转": ("BOOLEAN", {
                    "default": False,
                    "label_on": "反转 (宽↔高)",
                    "label_off": "正常"
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("宽度", "高度")
    FUNCTION = "process"
    CATEGORY = "凤希AI/工具"

    def process(self, 尺寸设置, 预设, 宽度, 高度, 反转):
        if not 尺寸设置:
            w, h = 1024, 1024
            parts = 预设.split()
            size_text = parts[-1]
            
            try:
                w_str, h_str = size_text.split("×")
                w = int(w_str.strip())
                h = int(h_str.strip())
            except (ValueError, IndexError):
                w, h = 1024, 1024
        else:
            w = 宽度
            h = 高度
        
        if 反转:
            return (h, w)
        return (w, h)