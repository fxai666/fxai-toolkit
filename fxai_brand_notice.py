# -*- coding: utf-8 -*-
"""凤希AI - 信息公示中心节点（自包含）。

节点本体直接展示品牌横幅（图 + 群号/官网/微信/公告等连接信息），
所有显示内容都写死在前端扩展 fxai_brand_notice.js 里，
不依赖 ComfyUI/input/ 下的任何图片文件，也不需要连线。

别人装了 fxai-toolkit 就有这个节点，横幅图随节点走，永远不会因为
工作流目录里缺图而红节点 —— 这正是替代旧版 LoadImage 品牌条的原因。

无输入、无输出、OUTPUT_NODE；version widget 仅供内部追踪横幅版本。
"""
from __future__ import annotations


class FxAiBrandNotice:
    VERSION = "1.13"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 隐藏标记：前端 JS 会把它藏起来；仅用于以后换横幅时对齐版本。
                "横幅版本": ("STRING", {"default": cls.VERSION}),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "noop"
    OUTPUT_NODE = True
    CATEGORY = "凤希AI/信息"

    def noop(self, 横幅版本=VERSION):
        return {"ui": {}}
