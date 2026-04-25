import torch
import comfy.samplers
import comfy.sample
import comfy.model_management

def get_ltx23_sigmas(model, steps):
    if steps == 4:
        return torch.tensor([14.6146, 6.4746, 1.9074, 0.4255, 0.0])
    elif steps == 6:
        return torch.tensor([14.6146, 9.4268, 6.4746, 3.9665, 1.9074, 0.4255, 0.0])
    elif steps == 8:
        return torch.tensor([14.6146, 11.6742, 9.4268, 7.5958, 6.0729, 3.9665, 1.9074, 0.6472, 0.0])
    try:
        sampling = model.model.model_sampling
    except AttributeError:
        try:
            sampling = model.model_sampling
        except AttributeError:
            sampling = model
    return comfy.samplers.calculate_sigmas(sampling, "exponential", steps).cpu()

class FxAiLTXCollector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "steps": ("INT", {"default": 6, "min": 4, "max": 8}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                "add_noise": ("BOOLEAN", {"default": False}),
                "sampler_name": (["euler", "euler_ancestral"],),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "凤希AI"

    def run(self, model, positive, negative, latent, steps, cfg, seed, add_noise, sampler_name):
        device = comfy.model_management.get_torch_device()
        av_latent = latent["samples"].clone().to(device)
        
        # 处理 seed
        if seed == -1:
            seed = torch.randint(0, 0xFFFFFFFFFFFFFFFF, (1,)).item()
        
        # ===================== 【正确初始化噪声】 =====================
        if add_noise:
            # 方式A：从零生成随机噪声（适用于 txt2vid 或需要随机起始）
            noise = comfy.sample.prepare_noise(av_latent, seed, None)
        else:
            # 方式B：零噪声（适用于 img2vid / AV 特征提取 / 确定性采集）
            noise = torch.zeros_like(av_latent, device=device)
        
        # 确保 noise 在正确的设备上
        noise = noise.to(device)
        
        sigmas = get_ltx23_sigmas(model, steps).to(device)
        sampler_obj = comfy.samplers.sampler_object(sampler_name)
        
        # 确保 sigmas 长度与 steps 匹配
        # 采样器期望 sigmas 长度 = steps + 1
        if len(sigmas) != steps + 1:
            sigmas = sigmas[:steps + 1]
        
        try:
            output = comfy.samplers.sample(
                model,
                noise,
                positive,
                negative,
                cfg,
                device,
                sampler_obj,
                sigmas,
                latent_image=av_latent,
                seed=seed
            )
        except Exception as e:
            # 如果采样失败，尝试不传 latent_image 参数
            print(f"First sampling attempt failed: {e}")
            output = comfy.samplers.sample(
                model,
                noise,
                positive,
                negative,
                cfg,
                device,
                sampler_obj,
                sigmas,
                seed=seed
            )
        
        # 将输出移回 CPU
        if isinstance(output, torch.Tensor):
            output = output.cpu()
        
        return ({"samples": output},)