import json

class FxAiMultiLineText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "索引": ("INT", {"default": 0, "min": 0, "forceInput": True}),
                "分段循环复用": ("INT", {"default": 1, "min": 1,"tooltip":"当大于1时，只会在设置的这个值的区域内来回循环分段提示词"}),
            },
            "optional":{
                "通用提示词": ("STRING", {"default": "", "forceInput": True}),
                "尾部通用提示词": ("STRING", {"default": "", "forceInput": True}),
                "lines_data": ("STRING", {"default": "[]", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "INT", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("索引提示词", "分段时长", "总行数", "索引", "音频索引", "转场")
    FUNCTION = "execute"
    CATEGORY = "凤希AI"

    def execute(self, 索引, 分段循环复用, 通用提示词="", 尾部通用提示词="", lines_data="[]"):
        分段循环复用 = 1 if (not isinstance(分段循环复用, int) or 分段循环复用 < 1) else 分段循环复用
        
        # 解析数据
        try:
            if isinstance(lines_data, str):
                lines = json.loads(lines_data.strip())
            elif isinstance(lines_data, (list, dict)):
                lines = lines_data
            else:
                lines = []
        except Exception as e:
            lines = []
            print(f"解析lines_data失败: {e}")

        total_count = len(lines)

        duration = 5.0
        audio_no = 0
        transition = True
        final_prompt = f"{通用提示词}{尾部通用提示词}"

        # 空数据直接返回默认值
        if total_count == 0 or 索引 < 0 or 索引 >= total_count:
            return (final_prompt, duration, total_count, 索引, audio_no, transition)

        # 循环复用
        if 分段循环复用 > 1:
            索引 = 索引 % 分段循环复用

        item = lines[索引]

        if isinstance(item, (list, tuple)):
            # 时长
            try:
                if len(item) >= 1:
                    duration = float(item[0])
            except:
                pass

            # 文本
            text = str(item[1]) if len(item) >= 2 else ""

            # 音频索引
            try:
                if len(item) >= 3:
                    audio_no = max(int(item[2]), 0)
            except:
                pass

            # 转场
            try:
                if len(item) >= 4:
                    transition = int(item[3]) >= 1
            except:
                pass
        else:
            # 纯文本
            text = str(item)

        # 最终提示词
        final_prompt = f"{通用提示词}{text}{尾部通用提示词}"
        return (final_prompt, duration, total_count, 索引, audio_no, transition)