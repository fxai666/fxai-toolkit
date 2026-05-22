import os
import platform
import subprocess

class FxAiAutoShutdown:
    """
    凤希AI - 秒后自动关机
    工作流执行完成后，等待指定秒数自动关机
    支持Windows / Linux / Mac 自动识别
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "触发输入": ("*",),  # 任意节点连接过来，执行完才关机
                "等待秒数": ("INT", {
                    "default": 60,
                    "min": 5,
                    "max": 86400,
                    "step": 1,
                    "display": "number"
                }),
                "启用关机": ("BOOLEAN", {
                    "default": True,
                    "label_on": "启用",
                    "label_off": "关闭"
                }),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("触发输出",)
    FUNCTION = "run"
    CATEGORY = "凤希AI/工具"
    OUTPUT_NODE = True

    def run(self, 触发输入, 等待秒数, 启用关机):
        if not 启用关机:
            print("✅ 凤希AI：自动关机已关闭")
            return (触发输入,)

        sys = platform.system()
        try:
            if sys == "Windows":
                # Windows 关机命令：shutdown -s -t 秒数
                subprocess.run(["shutdown", "-s", "-t", str(等待秒数)], capture_output=True)
                print(f"✅ 凤希AI：已设置 {等待秒数} 秒后自动关机")

            elif sys in ("Linux", "Darwin"):
                # Linux / Mac 用分钟计算，自动换算
                minutes = max(1, (等待秒数 + 59) // 60)
                subprocess.run(["sudo", "shutdown", "-h", f"+{minutes}"], capture_output=True)
                print(f"✅ 凤希AI：已设置 {minutes} 分钟后自动关机")

        except Exception as e:
            print(f"❌ 凤希AI：自动关机失败 - {str(e)}")

        return (触发输入,)