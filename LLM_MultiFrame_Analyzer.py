from PIL import Image
import numpy as np
from comfy.llm import get_llm_model

def tensor_to_pil(tensor):
    return Image.fromarray((tensor.cpu().numpy() * 255).astype(np.uint8))

class LLM_MultiFrame_Analyzer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # 自动加载 models/llm/ 下所有模型
                "llm_model": (get_llm_model(),),
                
                "images": ("IMAGE",),  # 首帧+尾帧，多张图
                "system_prompt": ("STRING", {"multiline": True}),
                "user_prompt": ("STRING", {"multiline": True}),
                "temperature": ("FLOAT", {"default": 0.7, "min":0, "max":1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "💊康复UI/LLM"

    def process(self, llm_model, images, system_prompt, user_prompt, temperature):
        # --------------------------
        # 官方标准解包模型
        # --------------------------
        model, _ = llm_model

        # 转图片
        pil_images = [tensor_to_pil(images[i]) for i in range(images.shape[0])]

        # --------------------------
        # 官方标准多模态消息格式
        # --------------------------
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                *[{"type": "image", "image": img} for img in pil_images],
                {"type": "text", "text": user_prompt}
            ]}
        ]

        # --------------------------
        # 官方标准调用方式
        # --------------------------
        try:
            output = model.chat(
                messages,
                temperature=temperature,
                max_new_tokens=2048
            )
        except Exception as e:
            return (f"错误：{str(e)}",)

        return (output.strip(),)