import torch
import math
import comfy.utils
import node_helpers
#https://github.com/PixWizardry/ComfyUI_PixQwenImageEditEnhanced/blob/main/nodes.py
DEFAULT_SYS = "根据图片和用户指令编辑图像，保留原图合理内容"

class FxAiQwenEditEnhanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "USER_PROMPT": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "SYSTEM_TEXT": ("STRING", {"multiline": True, "default": DEFAULT_SYS}),
                "ASSIST_HEAD": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "vae": ("VAE",),
                "images": ("IMAGE",), # 沿用原图批量队列输入，单端口批量多张
            }
        }
    RETURN_TYPES = ("CONDITIONING","STRING")
    RETURN_NAMES = ("conditioning","final_prompt")
    FUNCTION = "encode"
    CATEGORY = "advanced/conditioning"

    def encode(self, clip, USER_PROMPT, SYSTEM_TEXT, ASSIST_HEAD, vae=None, images=None):
        images_vl = []
        ref_latents = []
        image_prompt = ""
        USER_PROMPT = USER_PROMPT.replace("\n"," ").strip()

        # 批量队列IMAGE，自动遍历所有批次图片
        if images is not None:
            batch_num = images.shape[0]
            for i in range(batch_num):
                one_img = images[i:i+1,...]
                samples = one_img.movedim(-1,1).contiguous()
                _,_,h,w = samples.shape
                max_side = max(h,w)
                # Qwen视觉输入统一384长边等比
                scale = 384 / max_side
                nw,nh = int(w*scale),int(h*scale)
                img_resize = comfy.utils.common_upscale(samples,nw,nh,"area","disabled")
                images_vl.append(img_resize.movedim(1,-1).contiguous())

                # VAE编码原图latent用于参考绑定
                if vae is not None:
                    target_px = 1024*1024
                    s_vae = math.sqrt(target_px/(h*w))
                    vw = round(w*s_vae/8)*8
                    vh = round(h*s_vae/8)*8
                    vae_img = comfy.utils.common_upscale(samples,vw,vh,"area","disabled")
                    lat = vae.encode(vae_img.movedim(1,-1))
                    ref_latents.append(lat)

                image_prompt += f"图{i+1}<|vision_start|><|image_pad|><|vision_end|>"

        # 动态拼接对话模板（动态letter）
        dyn_template = f"<|im_start|>system\n{SYSTEM_TEXT}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n{ASSIST_HEAD}"
        user_full = image_prompt + USER_PROMPT
        final_prompt = dyn_template.format(user_full)

        # 正确视觉入参vision_images
        tokens = clip.tokenize(user_full, vision_images=images_vl, llama_template=dyn_template)
        cond = clip.encode_from_tokens_scheduled(tokens)

        # 挂载VAE参考潜变量
        if len(ref_latents)>0:
            cond = node_helpers.conditioning_set_values(cond,{"reference_latents":ref_latents},append=True)

        return (cond, final_prompt)