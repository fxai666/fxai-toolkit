import datetime
class FxAIAudioSegmentLoad:
    """
    音频分段截取工具（完整对齐+补最后一段差值逻辑）
    完全复刻原始的7步对齐规则：
    1. 分段时长累加 = 总时长
    2. 理论总帧数 = 总时长 × 帧率
    3. 每段向下【帧数对齐基数】对齐后累加 = 对齐总帧数
    4. 差值 = 理论总帧数 - 对齐总帧数
    5. 差值直接加到最后一段
    6. 最后一段再【帧数对齐基数】对齐
    7. 开始索引前正常计算
    """
    CATEGORY = "凤希AI/音频"
    FUNCTION = "extract_audio_segment"

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("剪切音频", "生成帧数")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 严格按你要求的顺序排列
                "帧率": ("INT", {"default": 24, "min": 1}),
                "当前索引": ("INT", {"default": 0, "min": 0}),
                "帧数对齐基数": ("INT", {"default": 8, "min": 1}),
                "过渡帧数": ("INT", {"default": 1, "min": 0}),

                # 原始输入保持不动
                "分段时长列表": ("LIST", {"forceInput": True}),
                "原始音频": ("AUDIO", {"forceInput": True}),
            },
        }

    # 向下对齐（返回纯整数）
    def align_down(self, frames, base):
        return int(frames // base * base)

    # 向上对齐（返回纯整数）
    def align_up(self, frames, base):
        if frames <= 0:
            return 0
        return int(((frames + base - 1) // base) * base)

    def extract_audio_segment(self, 帧率, 当前索引, 帧数对齐基数, 过渡帧数, 分段时长列表, 原始音频):
        print(f"✅ [凤希AI] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 开始渲染第 {当前索引+1} 个场景")
        # 1. 基础数据转换
        分段时长 = [float(s) for s in 分段时长列表]
        分段数量 = len(分段时长)

        if 分段数量 == 0:
            return (原始音频, int(过渡帧数))

        # 最后一段索引
        结束索引 = 分段数量 - 1

        # ==========================
        # 索引合法性校验
        # ==========================
        if 当前索引 < 0 or 当前索引 > 结束索引:
            raise ValueError(f"❌ 当前索引({当前索引}) 超出有效范围！允许范围：0 ~ {结束索引}")

        # --------------------------
        # 核心：原逻辑完整补齐最后一段（全程整数）
        # --------------------------
        # 理论原始帧数（转整数）
        分段原始帧数 = [int(时长 * 帧率) for 时长 in 分段时长]
        # 向下对齐
        分段对齐帧数 = [self.align_down(f, 帧数对齐基数) for f in 分段原始帧数]

        # 总理论帧数（整数）
        总理论帧数 = int(sum(分段时长) * 帧率)
        总对齐帧数 = sum(分段对齐帧数)
        缺失帧数 = 总理论帧数 - 总对齐帧数

        # 差值加到最后一段，并向上对齐（整数）
        分段对齐帧数[结束索引] = self.align_up(分段对齐帧数[结束索引] + 缺失帧数, 帧数对齐基数)
        分段时长[结束索引] = 分段对齐帧数[结束索引] / 帧率
        # --------------------------
        # 计算当前分段的开始秒数
        # --------------------------
        前面总对齐帧数 = sum(分段对齐帧数[:当前索引])
        实际开始秒 = 前面总对齐帧数 / 帧率  # 这里是秒数，允许浮点

        # --------------------------
        # ComfyUI 音频截取（兼容1/2/3维，100%保留你的计算逻辑）
        # --------------------------
        print(f"原始音频{原始音频}")
        sample_rate = 原始音频["sample_rate"]
        waveform = 原始音频["waveform"]

        # ==============================================
        # ✅ 核心修复：兼容 1维 / 2维 / 3维 任意音频格式
        # ==============================================
        total_samples = waveform.size(-1)  # 永远取最后一维 = 总采样数（通用所有维度）

        # ========== 100% 保留你原来的计算逻辑 ==========
        start_sample = int(实际开始秒 * sample_rate)
        end_sample = start_sample + int(分段时长[当前索引] * sample_rate)

        # ========== 安全边界（修复空音频问题） ==========
        start_sample = max(0, min(start_sample, total_samples))
        end_sample = max(start_sample, min(end_sample, total_samples))

        # ========== 切片：自动兼容所有维度，不破坏格式 ==========
        截取后音频_data = waveform[..., start_sample:end_sample]  # ... 代表所有前面维度，通用1/2/3维
        截取后音频 = {
            "waveform": 截取后音频_data,
            "sample_rate": sample_rate
        }

        # ==========================
        # 最终帧数：硬整形保证！
        # 对齐帧数(8的倍数) + 过渡帧数 = 绝对整数
        # ==========================
        生成帧数 = int(分段对齐帧数[当前索引] + 过渡帧数)

        return (截取后音频, 生成帧数)