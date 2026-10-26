# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
"""凤希AI - H3 多卡并行：把一个模型切成几份，几卡一起算。

和 `SelectModelDevice` 不是一回事。那个是**分卡** —— 整个模型搬到某一张卡上，
卡与卡之间各干各的。这个是**多卡** —— 同一个模型按头 / 按 ffn 切分，每卡存一份
切片，每个 block 用 all_reduce 合一次，速度按算力叠加。

决策不靠用户选版本，靠 `decide_from_model` 按实际权重体积和每卡空闲显存算：
卡数从少到多试，装不下就把「每步只算一次」的模块（adaln / refiner）卸到 CPU。
"""
from __future__ import annotations

import logging

import torch

import comfy.model_management as mm

from .devices import decide_from_model
from .runner import install_parallel

__all__ = ["FxAiH3MultiGPU"]


def _find_h3(model):
    """从 ModelPatcher 往下找到 MiniMaxH3Model。"""
    m = getattr(model, "model", None)
    for _ in range(12):
        if m is None:
            return None
        if type(m).__name__ == "MiniMaxH3Model":
            return m
        for attr in ("model", "inner_model", "diffusion_model", "unet_model"):
            nxt = getattr(m, attr, None)
            if nxt is not None:
                m = nxt
                break
        else:
            return None
    return None


class FxAiH3MultiGPU:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": ("MODEL",),
                "模式": (["自动", "手动指定"], {
                    "default": "自动",
                    "tooltip": "自动：按每张卡的空闲显存和带宽自己排，卡数从少到多试。\n"
                               "手动指定：严格按下面填的卡号顺序切分，第一位是 0 号分片。"}),
                "卡号": ("STRING", {
                    "default": "0",
                    "tooltip": "只有「手动指定」才读，自动模式下这个值被完全忽略（填什么都行）。\n"
                               "写法 0,1 / 1,0 / 0123，也可以 0 1 或 0;1。顺序就是分片顺序，不会重排。"}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("模型", "说明")
    FUNCTION = "patch"
    CATEGORY = "凤希AI/多卡"

    def patch(self, 模型, 模式, 卡号):
        inner = _find_h3(模型)
        if inner is None:
            raise ValueError("FxAiH3MultiGPU 只支持 MiniMax H3 模型")

        done = getattr(inner, "_fxai_parallel", None)
        if done is not None:
            return (模型, f"已装过：{done}")

        # 自动模式不读卡号 —— 这里就把它丢掉，免得下游谁多看一眼又被默认值 0 带偏。
        manual = (模式 == "手动指定")

        # **决策前必须把整张卡腾干净**。节点执行时 UNETLoader 已经把模型加载过一次，
        # 卡 0 上残留几个 G；此时量空闲显存会把这部分当成「别人占的」，
        # 自动模式就会误判装不下（实测 19.5G 模型报「每卡 12.98G > 可用 9.67G」）。
        # 切分本来就要先把权重落内存，顺序调过来正好一举两得。
        n_dev = torch.cuda.device_count()
        free_before = {i: torch.cuda.mem_get_info(i)[0] for i in range(n_dev)}
        inner.to("cpu")
        torch.cuda.empty_cache()

        # 只撤自己还不够。文本编码器（约 15G）和两路 VAE（各约 5G）在进这个节点之前
        # 编条件时就已经上卡，算完不会自己退出，卡 0 会长期被它们占着；自动模式按
        # min(各卡可用) 等分，用这个被占满的数字去算，必然判「装不下」。
        # 这些模型之后按需重载，卸掉不影响后续，所以这里连同它们一起腾。
        keep = [lm for lm in mm.current_loaded_models if lm.model is 模型]
        for i in range(n_dev):
            dev = torch.device(f"cuda:{i}")
            mm.free_memory(mm.get_total_memory(dev), dev, keep_loaded=keep)
        torch.cuda.empty_cache()

        freed = [(i, (torch.cuda.mem_get_info(i)[0] - free_before[i]) / (1 << 30))
                 for i in range(n_dev)]
        freed = [(i, g) for i, g in freed if g > 0.05]

        ids, placement, notes, groups, devs = decide_from_model(
            inner, mode=模式, spec=(卡号 if manual else ""))
        if freed:
            notes.append("决策前腾出显存：" +
                         "、".join(f"cuda:{i} +{g:.2f}G" for i, g in freed))

        if len(ids) < 2:
            if manual:
                lines = notes + [
                    f"手动指定只填了 1 张卡（cuda:{ids[0]}），按单卡整份跑，不切分。",
                    "要双卡请在卡号里写 0,1",
                ]
            else:
                # 自动模式下这是正常结果：模型装得下，没必要为了并行硬占两张卡
                lines = notes + [f"自动判定：cuda:{ids[0]} 一张卡装得下，不做切分"]
            # 权重已经被我们搬到内存了，放回卡上再交回去，免得后续采样多搬一次
            inner.to(f"cuda:{ids[0]}")
            return (模型, "\n".join(lines))

        runner = install_parallel(inner, ids, placement)
        runner.free_originals()
        inner._fxai_parallel = runner

        lines = notes
        lines.append(f"每卡常驻 {placement.per_card_gib:.2f}G 权重")
        lines.extend(runner.notes)
        text = "\n".join(lines)
        logging.info("[凤希AI] H3 多卡并行：\n" + text)
        return (模型, text)
