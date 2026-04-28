from typing import List, Dict, Any

class FxAiMultiPromptLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "多提示词数据": ("LIST", {"forceInput": True}),
                "索引值": ("INT", {"forceInput": True}),
                "帧率": ("INT", {"default": 24, "min": 1, "step": 1}),
                "默认提示词": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "刷新标记": ("INT", {"forceInput": True}),
                "通用提示词": ("STRING", {"default": "", "forceInput": True}),
                "尾部通用提示词": ("STRING", {"default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("最终提示词",)
    FUNCTION = "load_prompt"
    CATEGORY = "凤希AI/场景管理"

    def load_prompt(
        self,
        多提示词数据: List[Dict[str, Any]],
        索引值: int,
        帧率: int,
        默认提示词: str,
        刷新标记=0,
        通用提示词="",
        尾部通用提示词=""
    ) -> tuple[str]:
        try:
            if not 多提示词数据:
                return (默认提示词,)

            matched_items = [i for i in 多提示词数据 if i.get("索引编号") == 索引值]
            segment_parts = []

            # 正常循环生成，不做任何多余判断
            for item in matched_items:
                start_frm = round(float(item.get("开始时间", 0.0)) * 帧率) + 1
                end_frm = round(float(item.get("结束时间", 15.0)) * 帧率)
                prompt = item.get("提示词文本", "").strip()
                if prompt:
                    segment_parts.append(f"[{start_frm}-{end_frm}] {prompt}")

            # ====================== 你的核心逻辑：循环外只执行一次 ======================
            if segment_parts and float(matched_items[0].get("开始时间", 0)) == 0.0:
                segment_parts[0] = "[0" + segment_parts[0][segment_parts[0].find('-'):]
            # ==========================================================================

            final_lines = []
            if 通用提示词.strip():
                final_lines.append(通用提示词)
            
            final_lines.extend(segment_parts)
            
            if 尾部通用提示词.strip():
                final_lines.append(尾部通用提示词)

            final_text = "\n".join(final_lines)
            return (final_text if final_text.strip() else 默认提示词,)

        except Exception as e:
            print(f"[凤希AI] 加载提示词失败: {e}")
            return (默认提示词,)