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

    RETURN_TYPES = ("STRING", "STRING", "DICT")
    RETURN_NAMES = ("时序分行提示词", "分段纯文本提示词", "字典数据结构")
    FUNCTION = "load_prompt_data"
    CATEGORY = "凤希AI/场景管理"

    def load_prompt_data(
        self,
        多提示词数据: List[Dict[str, Any]],
        索引值: int,
        帧率: int,
        默认提示词: str,
        刷新标记=0,
        通用提示词="",
        尾部通用提示词=""
    ) -> tuple[str, str, dict]:
        try:
            if not 多提示词数据:
                return (默认提示词, 默认提示词, {
                    "global_prompt": 通用提示词 + " " + 尾部通用提示词,
                    "segments": [],
                    "total_frames": 0
                })

            # ==============================================
            # ✅ 第一步：筛选 + 预处理帧（只算一次！）
            # ==============================================
            matched_items = []
            for item in 多提示词数据:
                if item.get("索引编号") != 索引值:
                    continue

                # 预处理：秒 → 帧（只在这里算一次）
                start_frame = round(float(item.get("开始时间", 0.0)) * 帧率)
                end_frame = round(float(item.get("结束时间", 15.0)) * 帧率)
                prompt = item.get("提示词文本", "")

                if not prompt:
                    continue

                matched_items.append({
                    "prompt": prompt,
                    "start": start_frame,
                    "end": end_frame
                })

            # ==============================================
            # ✅ 第二步：自动修正断帧（上一段尾 == 下一段首 → +1）
            # ==============================================
            if len(matched_items) >= 2:
                for i in range(1, len(matched_items)):
                    prev_end = matched_items[i-1]["end"]
                    curr_start = matched_items[i]["start"]
                    if prev_end == curr_start:
                        matched_items[i]["start"] = curr_start + 1

            # ==============================================
            # ✅ 第三步：直接用预处理好的数据（无重复计算）
            # ==============================================
            segment_parts = []
            segment_prompts = []
            total_frames = 0

            for seg in matched_items:
                s = seg["start"]
                e = seg["end"]
                p = seg["prompt"]

                segment_parts.append(f"[{s}-{e}]:{p}")
                segment_prompts.append(p)
                if e > total_frames:
                    total_frames = e

            # 拼接时序提示词
            final_lines = []
            if 通用提示词:
                final_lines.append(通用提示词)
            if 尾部通用提示词:
                final_lines.append(尾部通用提示词)
            final_lines.extend(segment_parts)
            final_text = "\n".join(final_lines)
            if not final_text:
                final_text = 默认提示词

            # 拼接纯提示词
            global_parts = []
            if 通用提示词:
                global_parts.append(通用提示词)
            global_parts.extend(segment_prompts)
            if 尾部通用提示词:
                global_parts.append(尾部通用提示词)
            final_global_str = "\n".join(global_parts)
            if not final_global_str:
                final_global_str = 默认提示词

            # 输出字典
            global_prompt_str = (通用提示词 + " " + 尾部通用提示词)
            result = {
                "global_prompt": global_prompt_str,
                "segments": matched_items,  # 直接用预处理好的数组
                "total_frames": total_frames
            }

            return (final_text, final_global_str, result)

        except Exception as e:
            print(f"[凤希AI] 加载提示词失败: {e}")
            empty_result = {
                "global_prompt": 通用提示词 + " " + 尾部通用提示词,
                "segments": [],
                "total_frames": 0
            }
            return (默认提示词, 默认提示词, empty_result)