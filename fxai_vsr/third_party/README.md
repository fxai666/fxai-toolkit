# 第三方组件说明（Third-Party Notice）

本目录下的代码**不是**凤希AI原创，也不含任何 GPL 代码。

## third_party/sparse_sage/

- 来源：SageAttention 团队
  https://github.com/jt-zhang/Sparse_SageAttention_API
- 许可：**Apache License 2.0**（见同目录 `LICENSE.txt`）
- 内容：块稀疏 int8 注意力的 Triton 实现（`core.py` / `quant_per_block.py` /
  `sparse_int8_attn.py`）
- 本项目的用法：仅作为运行时依赖被 `fxai_vsr/ops.py` 导入，未修改源码。

Apache-2.0 与 MIT 兼容，保留版权声明与许可文本即可。

## 权重来源

本包**只读取公开权重文件，不含任何第三方源码**。权重由 FlashVSR 项目发布
（魔塔 `AI-ModelScope/FlashVSR`），共四个文件：

| 文件 | 内容 |
|---|---|
| `diffusion_pytorch_model_streaming_dmd.safetensors` | DiT，825 张量，fp32 |
| `LQ_proj_in.ckpt` | LQ 投影，8 键，bf16 |
| `TCDecoder.ckpt` | 解码器，66 键，fp32 |
| `posi_prompt.pth` | 固定文本上下文 (1,512,4096) |

放 `models/FlashVSR/`（目录名沿用上游品牌，权重是他们训练的，我们只读）。

## 为什么没有沿用上游的代码

`ComfyUI-FlashVSR_Ultra_Fast` 仓库整体是 **GPL-3.0**。其中的 DiT
（`wan_video_dit.py`）、解码器（`TCDecoder.py`）、流式管线（`pipelines/`）、
`utils.py`、`model_manager.py` 都属于 GPL-3 覆盖范围，不能并入本 MIT 项目。

因此 `fxai_vsr/` 下的文件（`ops.py` / `dit.py` / `lq.py` / `tcdecoder.py` /
`loader.py` / `pipeline.py`）是按论文描述与权重张量形状**重新实现**的独立代码，
与上游源码无继承关系，仅共享同一套权重。
