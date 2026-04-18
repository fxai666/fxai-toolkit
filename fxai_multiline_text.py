import json

class FxAiMultiLineText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "索引": ("INT", {"default": 0, "min": 0, "forceInput": True}),
            },
            "optional":{
                "通用提示词": ("STRING", {
                    "default": "",
                    "forceInput": True
                }),
                "尾部通用提示词": ("STRING", {
                    "default": "",
                    "forceInput": True
                }),
                "lines_data": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                }),
            }
        }
    
    # 新增音频索引返回值，调整返回类型定义
    RETURN_TYPES = ("STRING", "FLOAT", "INT", "INT", "INT")
    RETURN_NAMES = ("索引提示词", "分段时长", "总行数", "索引", "音频索引")
    FUNCTION = "execute"
    CATEGORY = "凤希AI"

    # 修复：参数顺序（带默认值放最后）+ 中文变量兼容
    def execute(self, 索引, 通用提示词="", 尾部通用提示词="", lines_data="[]"):
        try:
            # 修复：统一解析 lines_data，支持 str / list
            if isinstance(lines_data, str):
                lines = json.loads(lines_data.strip())
            elif isinstance(lines_data, (list, dict)):
                lines = lines_data
            else:
                lines = []
        except Exception as e:
            # 解析失败返回空列表
            lines = []
            print(f"解析lines_data失败: {e}")

        total_count = len(lines)
        default_sec = 5.0
        default_audio_no = 0  # 音频索引默认值
        
        # 空数据直接返回（包含默认音频索引）
        if total_count == 0:
            return ("", default_sec, 0, 索引, default_audio_no)
        
        # 索引越界判断
        if 索引 < 0 or 索引 >= total_count:
            return ("", default_sec, total_count, 索引, default_audio_no)

        # 取出当前行数据
        item = lines[索引]
        # 兼容格式：
        # 1. [时长, 文本, 音频索引] 完整格式
        # 2. [时长, 文本] 无音频索引格式
        # 3. 纯文本格式
        if isinstance(item, (list, tuple)):
            # 解析时长
            try:
                duration = float(item[0]) if len(item) >= 1 else default_sec
            except:
                duration = default_sec
            # 解析文本
            text = str(item[1]) if len(item) >= 2 else ""
            # 解析音频索引（新增）
            try:
                audio_no = int(item[2]) if len(item) >= 3 else default_audio_no
                # 确保音频索引≥1
                audio_no = max(audio_no, 0)
            except:
                audio_no = default_audio_no
        else:
            # 纯文本格式
            text = str(item)
            duration = default_sec
            audio_no = default_audio_no

        # 拼接最终提示词
        final_prompt = f"{通用提示词}{text}{尾部通用提示词}"
        
        # 返回值新增音频索引，与RETURN_TYPES/RETURN_NAMES对应
        return (final_prompt, duration, total_count, 索引, audio_no)