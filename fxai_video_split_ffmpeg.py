import json
import os
import subprocess

class FxAiVideoSplitFFmpeg:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "视频路径": ("STRING", {"default": "","tooltip": "视频文件完整地址"}),
                "分割秒数列表": ("STRING", {"default": "600,900", "tooltip": "逗号分隔浮点数/整数秒数，如600,900，代表0~600、600~900、900至结尾三段视频；超过视频总时长的数值会自动丢弃"}),
            },
            "optional": {
                "封面图片路径": ("STRING", {"default": "", "tooltip": "仅支持单张图片文件；为空则不开启片头，维持纯分割"}),
                "前置帧数片头": ("INT", {"default": 1, "min": 1, "tooltip": "第2段及往后片段开头追加图片片头，图片等比例居中；先播放图片再播放视频"}),
            }
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
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
            if res.returncode == 0 and res.stdout.strip():
                return float(res.stdout.strip())
            return None
        except Exception:
            return None

    def _get_video_fps(self, video_path):
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
            fps_str = res.stdout.strip()
            if "/" in fps_str:
                num, den = fps_str.split("/")
                return float(num)/float(den)
            return float(fps_str)
        except Exception:
            return 24.0

    def _get_video_resolution(self, video_path):
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
            lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
            w = int(lines[0])
            h = int(lines[1])
            return (w, h)
        except Exception:
            return (1920, 1080)

    def _build_global_head(self, src_video, img_path, out_head, dur_sec, vid_w, vid_h, fps):
        """生成片头：内部自动读取源视频音频参数，生成匹配规格静音"""
        cmd_probe = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channel_layout",
            "-of", "default=noprint_wrappers=1:nokey=1",
            src_video
        ]
        res_probe = subprocess.run(cmd_probe, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=8)
        probe_lines = [x.strip() for x in res_probe.stdout.strip().splitlines() if x.strip()]
        if len(probe_lines) >= 2:
            audio_sr = probe_lines[0]
            audio_layout = probe_lines[1]
        else:
            audio_sr = "44100"
            audio_layout = "stereo"

        # 关键：直接读取【原视频真实视频流timebase】，不要自己随便计算tbn！
        cmd_tbn = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=time_base",
            "-of", "default=noprint_wrappers=1:nokey=1",
            src_video
        ]
        res_tbn = subprocess.run(cmd_tbn, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=8)
        time_base_str = res_tbn.stdout.strip()
        if "/" in time_base_str:
            tb_num, tb_den = time_base_str.split("/")
        else:
            tb_num = "1"
            tb_den = str(int(fps * 2))

        cmd = ["ffmpeg", "-y"]
        cmd.extend(["-loop", "1", "-i", img_path])
        cmd.extend(["-t", str(dur_sec)])

        filter_complex = (
            f"[0:v]fps={fps},"
            f"scale=w={vid_w}:h={vid_h}:force_original_aspect_ratio=decrease,"
            f"pad={vid_w}:{vid_h}:(ow-iw)/2:(oh-ih)/2:color=black[vout];"
            f"anullsrc=channel_layout={audio_layout}:sample_rate={audio_sr}[aout]"
        )
        cmd.extend(["-filter_complex", filter_complex])
        cmd.extend(["-map", "[vout]", "-map", "[aout]"])
        # 强制使用源视频完全一致的timebase，根除时基不匹配
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-video_track_timescale", tb_den
        ])
        cmd.extend(["-c:a", "aac"])
        cmd.append(out_head)
        return cmd

    def _cut_single_segment(self, src_video, start, end, out_temp):
        """截取源视频对应区间，c copy高速输出临时片段"""
        cmd = ["ffmpeg", "-y", "-ss", str(start)]
        if end is not None:
            dur = end - start
            cmd.extend(["-t", str(dur)])
        cmd.extend(["-i", src_video, "-c", "copy", out_temp])
        return cmd

    def _concat_head_and_seg(self, head_path, seg_path, out_final):
        """片头 + 视频片段 合并输出成品"""
        folder = os.path.dirname(out_final)
        list_txt = os.path.join(folder, "_tmp_concat_list.txt")
        with open(list_txt, "w", encoding="utf-8") as f:
            f.write(f"file '{head_path}'\n")
            f.write(f"file '{seg_path}'\n")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_txt,
            "-c", "copy",
            out_final
        ]
        return cmd, list_txt

    def main(self, 视频路径="", 分割秒数列表="", 封面图片路径="", 前置帧数片头=30):
        clean_video_path = 视频路径.strip().replace("\u202a", "").replace("\u202b", "")
        cover_img_path = 封面图片路径.strip().replace("\u202a", "").replace("\u202b", "")
        enable_head = bool(cover_img_path and os.path.isfile(cover_img_path))

        if not os.path.exists(clean_video_path):
            print(f"[FxAI视频分割] 错误：视频文件不存在 {clean_video_path}")
            return ("",)

        if enable_head and not os.path.isfile(cover_img_path):
            print(f"[FxAI视频分割] 警告：封面图片不存在，关闭片头功能")
            enable_head = False

        total_sec = self._get_video_total_seconds(clean_video_path)
        if total_sec is None:
            print("[FxAI视频分割] 错误：无法读取视频时长")
            return ("",)
        print(f"[FxAI视频分割] 视频总时长：{round(total_sec,2)} 秒")

        video_fps = self._get_video_fps(clean_video_path)
        vid_width, vid_height = self._get_video_resolution(clean_video_path)
        print(f"[FxAI视频分割] 分辨率 {vid_width}×{vid_height} FPS:{video_fps}")

        try:
            raw_split = [float(s.strip()) for s in 分割秒数列表.split(",") if s.strip()]
        except Exception:
            print("[FxAI视频分割] 分割秒数格式错误，请使用逗号分隔数字")
            return ("",)

        discard_points = [p for p in raw_split if p >= total_sec]
        valid_split_raw = [p for p in raw_split if p < total_sec]
        split_points = sorted(list(set(valid_split_raw)))
        if discard_points:
            print(f"[FxAI视频分割] 舍弃超出时长分割点：{discard_points}")
        print(f"[FxAI视频分割] 有效分割点位：{split_points}")

        file_dir = os.path.dirname(clean_video_path)
        file_name_raw, ext = os.path.splitext(os.path.basename(clean_video_path))
        split_folder = os.path.join(file_dir, file_name_raw)
        if not os.path.exists(split_folder):
            os.makedirs(split_folder)
        print(f"[FxAI视频分割] 输出目录：{split_folder}")

        time_segments = [0.0] + split_points + [None]
        global_head_path = os.path.join(split_folder, "_global_cover_head_tmp" + ext)
        head_duration = 前置帧数片头 / video_fps

        # =========【第一步：如果开启片头，只生成一次全局片头】=========
        if enable_head:
            print(f"[FxAI视频分割] 开始生成全局片头视频，时长 {round(head_duration,4)}s")
            cmd_head = self._build_global_head(
                src_video=clean_video_path,
                img_path=cover_img_path,
                out_head=global_head_path,
                dur_sec=head_duration,
                vid_w=vid_width,
                vid_h=vid_height,
                fps=video_fps
            )
            proc_head = subprocess.run(cmd_head, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc_head.returncode != 0:
                print(f"[FxAI视频分割] 全局片头生成失败：\n{proc_head.stderr}")
                return ("",)
            print(f"[FxAI视频分割] 全局片头生成完成：{global_head_path}")

        # =========【第二步：循环处理所有分段】=========
        for idx in range(len(time_segments)-1):
            seg_num = idx + 1
            start_sec = time_segments[idx]
            end_sec = time_segments[idx+1]
            final_out = os.path.join(split_folder, f"{file_name_raw}_{seg_num}{ext}")

            # 第一段 或者 不启用片头：直接截取输出成品
            if seg_num == 1 or not enable_head:
                cmd_cut = ["ffmpeg", "-y", "-ss", str(start_sec)]
                if end_sec is not None:
                    dur = end_sec - start_sec
                    cmd_cut.extend(["-t", str(dur)])
                cmd_cut.extend(["-i", clean_video_path, "-c", "copy", final_out])
                proc_cut = subprocess.run(cmd_cut, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if proc_cut.returncode != 0:
                    print(f"[FxAI视频分割] 分段{seg_num}切割失败\n{proc_cut.stderr}")
                    return ("",)
                print(f"[FxAI视频分割] 完成分段{seg_num} → {final_out}")
                continue

            # ===== 第2段及以后流程 =====
            temp_seg = os.path.join(split_folder, f"_temp_seg_{seg_num}{ext}")
            # 1. 临时截取本段视频区间
            cmd_cut_temp = self._cut_single_segment(clean_video_path, start_sec, end_sec, temp_seg)
            proc_temp = subprocess.run(cmd_cut_temp, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc_temp.returncode != 0:
                print(f"[FxAI视频分割] 分段{seg_num}临时截取失败\n{proc_temp.stderr}")
                return ("",)

            # 2. 全局片头 + 临时片段 合并输出最终文件
            cmd_concat, list_txt = self._concat_head_and_seg(global_head_path, temp_seg, final_out)
            proc_concat = subprocess.run(cmd_concat, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc_concat.returncode != 0:
                print(f"[FxAI视频分割] 分段{seg_num}合并失败\n{proc_concat.stderr}")
                return ("",)

            # 3. 立刻清理本段临时文件，不占用磁盘
            try:
                os.remove(temp_seg)
                os.remove(list_txt)
            except Exception:
                pass
            print(f"[FxAI视频分割] 完成带片头分段{seg_num} → {final_out}")

        # =========【收尾：删除全局片头临时文件】=========
        if enable_head and os.path.exists(global_head_path):
            try:
                os.remove(global_head_path)
            except Exception:
                pass

        print(f"[FxAI视频分割] 全部任务执行完毕，输出文件夹：{split_folder}")
        return (split_folder,)