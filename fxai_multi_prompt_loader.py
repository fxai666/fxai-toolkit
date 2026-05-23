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

    RETURN_TYPES = ("STRING", "STRING", "DICT", "STRING")
    RETURN_NAMES = ("时序按帧数提示词", "分段纯文本提示词", "字典数据结构", "时序按秒数提示词")
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
    ) -> tuple[str, str, dict, str]:
        try:
            if not 多提示词数据:
                empty_dict = {
                    "global_prompt": f"{通用提示词} {尾部通用提示词}".strip(),
                    "segments": [],
                    "total_frames": 0
                }
                return (默认提示词, 默认提示词, empty_dict, "")

            # ==============================================
            # ✅ 第一步：筛选 + 秒数+帧 统一预处理（只跑一次！）
            # ==============================================
            matched_items = []
            sec_format_lines = []  # 直接在这里收集秒数格式
            for item in 多提示词数据:
                if item.get("索引编号") != 索引值:
                    continue

                # 原始秒数（直接用，不转换）
                start_sec = float(item.get("开始时间", 0.0))
                end_sec = float(item.get("结束时间", 15.0))
                prompt = item.get("提示词文本", "")
                if not prompt:
                    continue

                # 帧计算
                start_frame = round(start_sec * 帧率)
                end_frame = round(end_sec * 帧率)

                matched_items.append({
                    "prompt": prompt,
                    "start": start_frame,
                    "end": end_frame
                })

                # ✅ 直接生成你要的秒数格式文本！一步到位
                sec_format_lines.append(f"[{start_sec}-{end_sec}s]".replace(".0", "") + f" {prompt} |")

            # ==============================================
            # ✅ 第二步：断帧修正
            # ==============================================
            if len(matched_items) >= 2:
                for i in range(1, len(matched_items)):
                    prev_end = matched_items[i-1]["end"]
                    curr_start = matched_items[i]["start"]
                    if prev_end == curr_start:
                        matched_items[i]["start"] = curr_start + 1

            # ==============================================
            # ✅ 第三步：拼接原有输出
            # ==============================================
            segment_parts = []
            segment_prompts = []
            total_frames = 0
            for seg in matched_items:
                s, e, p = seg["start"], seg["end"], seg["prompt"]
                segment_parts.append(f"[{s}-{e}]:{p}")
                segment_prompts.append(p)
                total_frames = max(total_frames, e)

            # 时序分行提示词
            final_lines = []
            if 通用提示词:
                final_lines.append(通用提示词)
            if 尾部通用提示词:
                final_lines.append(尾部通用提示词)
            final_lines.extend(segment_parts)
            final_text = "\n".join(final_lines) or 默认提示词

            # 分段纯文本提示词
            global_parts = []
            if 通用提示词:
                global_parts.append(通用提示词)
            global_parts.extend(segment_prompts)
            if 尾部通用提示词:
                global_parts.append(尾部通用提示词)
            final_global_str = "\n".join(global_parts) or 默认提示词

            # 秒数格式最终文本
            sec_format_text = "\n".join(sec_format_lines)

            # 字典
            global_prompt_str = f"{通用提示词} {尾部通用提示词}".strip()
            result = {
                "global_prompt": global_prompt_str,
                "segments": matched_items,
                "total_frames": total_frames
            }

            return (final_text, final_global_str, result, sec_format_text)

        except Exception as e:
            print(f"[凤希AI] 加载提示词失败: {e}")
            empty_dict = {
                "global_prompt": f"{通用提示词} {尾部通用提示词}".strip(),
                "segments": [],
                "total_frames": 0
            }
            return (默认提示词, 默认提示词, empty_dict, "")