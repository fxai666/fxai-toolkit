import json
import os
import subprocess

class FxAiVideoSplitFFmpeg:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "视频路径": ("STRING", {"default": ""}),
                "分割秒数列表": ("STRING", {"default": "600,900", "tooltip": "逗号分隔浮点数/整数秒数，如600,900，代表0~600、600~900、900至结尾三段视频；超过视频总时长的数值会自动丢弃"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("输出文件夹路径",)
    FUNCTION = "main"
    OUTPUT_NODE = True
    CATEGORY = "凤希AI/视频"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def _get_video_total_seconds(self, video_path):
        """调用ffprobe获取视频总时长（秒），失败返回None"""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            if res.returncode == 0 and res.stdout.strip():
                return float(res.stdout.strip())
            return None
        except Exception:
            return None

    def main(self, 视频路径="", 分割秒数列表=""):
        output_paths = []
        # 清洗路径：移除不可见特殊控制字符
        clean_video_path = 视频路径.strip().replace("\u202a", "").replace("\u202b", "")

        # 校验文件存在
        if not os.path.exists(clean_video_path):
            print(f"[FxAI视频分割] 错误：视频文件不存在 {clean_video_path}")
            return ("",)

        # 获取视频总时长
        total_sec = self._get_video_total_seconds(clean_video_path)
        if total_sec is None:
            print("[FxAI视频分割] 错误：无法读取视频时长，请确认ffmpeg/ffprobe已配置系统环境变量，视频文件正常")
            return ("",)
        print(f"[FxAI视频分割] 视频总时长：{round(total_sec,2)} 秒")

        # 解析用户输入分割秒数
        try:
            raw_split = [float(s.strip()) for s in 分割秒数列表.split(",") if s.strip()]
        except Exception:
            print("[FxAI视频分割] 错误：分割秒数格式错误，请使用逗号分隔数字，示例：600,900")
            return ("",)

        # 过滤：只保留 < 视频总时长 的分割点，丢弃超出时长的数值
        discard_points = [p for p in raw_split if p >= total_sec]
        valid_split_raw = [p for p in raw_split if p < total_sec]
        # 去重+升序
        split_points = sorted(list(set(valid_split_raw)))

        if discard_points:
            print(f"[FxAI视频分割] 已舍弃超出视频时长的分割点：{','.join([str(x) for x in discard_points])}")
        print(f"[FxAI视频分割] 有效分割节点：{','.join([str(x) for x in split_points]) if split_points else '无'}")

        # 拆分原文件名、目录、后缀
        file_dir = os.path.dirname(clean_video_path)
        file_name_raw, ext = os.path.splitext(os.path.basename(clean_video_path))
        # 新建分段文件夹：同级目录下 [原视频名]
        split_folder = os.path.join(file_dir, f"{file_name_raw}")
        # 不存在则创建目录
        if not os.path.exists(split_folder):
            os.makedirs(split_folder)
        print(f"[FxAI视频分割] 分段文件输出目录：{split_folder}")

        # 时间区间：0 + 有效分割点 + 末尾None
        time_segments = [0.0] + split_points + [None]

        # 循环分段
        for idx in range(len(time_segments) - 1):
            start_sec = time_segments[idx]
            end_sec = time_segments[idx + 1]
            # 文件生成在新建文件夹内
            out_name = f"{file_name_raw}_{idx+1}{ext}"
            out_full_path = os.path.join(split_folder, out_name)
            output_paths.append(out_full_path)

            cmd = ["ffmpeg", "-y", "-ss", str(start_sec)]
            if end_sec is not None:
                duration = end_sec - start_sec
                cmd.extend(["-t", str(duration)])
            cmd.extend(["-i", clean_video_path, "-c", "copy", out_full_path])

            try:
                subprocess.run(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"[FxAI视频分割] 成功生成分段{idx+1}：{out_full_path}")
            except Exception as e:
                print(f"[FxAI视频分割] 分段{idx+1}执行失败：{str(e)}")
                return ("",)

        print(f"[FxAI视频分割] 分段完成，共生成{len(output_paths)}个分段文件")
        # 直接输出文件夹路径
        return (split_folder,)