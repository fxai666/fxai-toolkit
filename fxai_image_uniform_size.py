import torch
from fxai_image_utils import ImageSizeController

class FxAiImageUniformSize:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{"输出宽度":("INT",{"default":704,"min":32,"max":8192,"step":32}),"输出高度":("INT",{"default":1280,"min":32,"max":8192,"step":8}),"图片序列":("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出序列",)
    FUNCTION = "process"
    CATEGORY = "凤希AI/图片"

    def process(self,输出宽度,输出高度,图片序列):
        ctrl=ImageSizeController(输出宽度,输出高度,(255,255,255))
        lst=图片序列 if isinstance(图片序列,list) else [图片序列]
        res=[]
        for t in lst:
            t=t.unsqueeze(0) if t.dim()==3 else t
            for f in t:res.append(ctrl.crop_fill_to_canvas(f[None])[0])
        return (torch.stack(res),)