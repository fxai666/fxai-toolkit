import torch
import comfy.samplers
import comfy.sample
from comfy import model_management

class FxAiLTX23Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "ref_latent": ("LATENT",),#首帧人物锚定图
                "audio_cond": ("CONDITIONING",),#LTX音频编码COND（LTXAudioConditioner输出）
                "steps": ("INT",{"default":12,"min":4,"max":32}),
                "cfg_scale": ("FLOAT",{"default":3.0,"min":1.0,"max":10.0}),
                "stg_weight": ("FLOAT",{"default":0.38,"min":0.0,"max":1.2}),
                "id_anchor_strength": ("FLOAT",{"default":0.35,"min":0.1,"max":0.8}),
                "audio_sync_weight": ("FLOAT",{"default":0.45,"min":0.1,"max":0.8})#音频口型同步权重
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample_id_fix"
    CATEGORY = "LTX23/自定义采样(音画+人物锁定)"

    def sample_id_fix(self,model,positive,negative,latent,ref_latent,audio_cond,steps,cfg_scale,stg_weight,id_anchor_strength,audio_sync_weight):
        device = model_management.get_torch_device()
        z_in = latent["samples"].clone()
        ref_z = ref_latent["samples"][0:1].to(device, dtype=z_in.dtype)
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), steps).to(device)
        prev_lat = [z_in.clone()]

        #合并音频条件到正向提示，LTX原生AV跨模态注意力识别音频特征{insert\_element\_1\_}
        pos_merge = comfy.conditioning.concat(positive,audio_cond)

        def sampler_callback(x, sigma, step):
            total = len(sigmas)-1
            ratio = step / total if total>0 else 0.0
            #1、人物ID锚定：前40%采样步强锁脸型五官
            id_w = id_anchor_strength * (1.0 if ratio < 0.4 else (1-ratio))
            x = x + id_w * (ref_z - x[:1])
            #2、STG帧间时序约束，抑制逐帧人物漂移
            if step>0:
                x = x + stg_weight*(prev_lat[0] - x)
            #3、音频时序引导：按音频特征修正画面，驱动口型同步，不破坏人物特征
            x = x + audio_sync_weight * torch.tanh(ref_z - x)
            prev_lat[0] = x.clone()
            return x

        sampler = comfy.samplers.get_sampler("euler")
        out_samples = comfy.sample.sample(
            model, z_in, steps, cfg_scale, sampler, pos_merge, negative,
            sigmas=sigmas, sampler_callback=sampler_callback
        )
        return ({"samples":out_samples},)