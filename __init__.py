import subprocess
import sys
import importlib
import os

def install_package(package):
    try:
        importlib.import_module(package)
        return True
    except ImportError:
        print(f"[凤希AI音频分段器] 未找到 {package}，正在自动安装...")
        python_exe = sys.executable
        try:
            subprocess.check_call([python_exe, "-m", "pip", "install", package])
            print(f"[凤希AI音频分段器] {package} 安装成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[凤希AI音频分段器] 安装 {package} 失败: {e}")
            return False

required_packages = ["pydub"]
for pkg in required_packages:
    install_package(pkg)

# 导入类（HTTP路由会在import时自动注册）
from .fxai_audio_segments import FxAiAudioSegmenter
from .fxai_multiline_text import FxAiMultiLineText
from .fxai_image_manager import FxAiImageManager
from .fxai_image_load import FxAiLoadImageByIndex
from .fxai_prompt_optimization import FxAiPromptGenerator
from .fxai_size_config import FxAiSizeConfig
from .fxai_multi_audio_load import FxAiMultiAudioLoad
from .fxai_audio_manager import FxAiAudioManager
from .fxai_audio_load import FxAiLoadAudioByIndex
from .fxai_prompt_manager import FxAiPromptManager
from .fxai_resize_image_downscale import FxAiImageDownscale
from .fxai_prompt_load import FxAiLoadPromptByIndex
from .fxai_scene_manager import FxAiSceneManager
from .fxai_scene_load import FxAiSceneLoad

# 统一注册
NODE_CLASS_MAPPINGS = {
    "FxAiAudioSegmenter": FxAiAudioSegmenter,
    "FxAiMultiLineText": FxAiMultiLineText,
	"FxAiImageManager":FxAiImageManager,
    "FxAiLoadImageByIndex": FxAiLoadImageByIndex,
	"FxAiPromptGenerator":FxAiPromptGenerator,
	"FxAiSizeConfig":FxAiSizeConfig,
	"FxAiMultiAudioLoad":FxAiMultiAudioLoad,
	"FxAiAudioManager":FxAiAudioManager,
	"FxAiLoadAudioByIndex":FxAiLoadAudioByIndex,
	"FxAiPromptManager":FxAiPromptManager,
	"FxAiImageDownscale":FxAiImageDownscale,
	"FxAiLoadPromptByIndex":FxAiLoadPromptByIndex,
	"FxAiSceneManager":FxAiSceneManager,
	"FxAiSceneLoad":FxAiSceneLoad,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FxAiAudioSegmenter": "凤希AI - 音频分段器",
    "FxAiSceneManager": "凤希AI - 场景管理V2 - 进Q群与更多的群友学习：775649071",
    "FxAiSceneLoad": "凤希AI - 场景数据加载",
    "FxAiMultiLineText": "凤希AI - 场景管理V1 - 进Q群与更多的群友学习：775649071",
    "FxAiImageManager": "凤希AI - 图像管理器 - 群友扫地僧友情开发",
    "FxAiLoadImageByIndex": "凤希AI - 图像管理器 - 图片加载",
	"FxAiPromptGenerator":"凤希AI - 提示词优化 - 本地Ollama",
	"FxAiSizeConfig":"凤希AI - 宽高设置",
	"FxAiMultiAudioLoad":"凤希AI - 多音频加载",
	"FxAiAudioManager":"凤希AI - 音频管理器",
	"FxAiLoadAudioByIndex":"凤希AI - 音频管理器 - 音频加载",
	"FxAiImageDownscale":"凤希AI - 图片缩小 - 按倍数",
	"FxAiPromptManager":"凤希AI - 提示词管理",
	"FxAiLoadPromptByIndex":"凤希AI - 提示词管理 - 提示词加载",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]