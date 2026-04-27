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
from .fxai_audio_segments_v2 import FxAiAudioSegmenterV2
from .fxai_audio_frame_load import FxAIAudioSegmentLoad
from .fxai_multiline_text import FxAiMultiLineText
from .fxai_multiline_text_load import FxAiMultiLineTextLoad
from .fxai_image_manager import FxAiImageManager
from .fxai_image_manager_v2 import FxAiImageManagerV2
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
from .fxai_scene_manager_v2 import FxAiSceneManagerV2
from .fxai_scene_load import FxAiSceneLoad
from .fxai_scene_load_v2 import FxAiSceneLoadV2
from .fxai_video_generator import FxAiVideoGenerator
from .fxai_video_merger import FxAiVideoMerger
from .fxai_video_preview import FxAiVideoPreview
from .fxai_video_manager import FxAiVideoManager
from .fxai_generator_controller import FxAIGeneratorController
from .fxai_frame_generator import FxAiFrameGenerator

# 统一注册
NODE_CLASS_MAPPINGS = {
    "FxAiAudioSegmenter": FxAiAudioSegmenter,
    "FxAiAudioSegmenterV2": FxAiAudioSegmenterV2,
    "FxAIAudioSegmentLoad": FxAIAudioSegmentLoad,
    "FxAiMultiLineText": FxAiMultiLineText,
    "FxAiMultiLineTextLoad": FxAiMultiLineTextLoad,
	"FxAiImageManager":FxAiImageManager,
	"FxAiImageManagerV2":FxAiImageManagerV2,
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
	"FxAiSceneManagerV2":FxAiSceneManagerV2,
	"FxAiSceneLoad":FxAiSceneLoad,
	"FxAiSceneLoadV2":FxAiSceneLoadV2,
	"FxAiVideoGenerator":FxAiVideoGenerator,
	"FxAiVideoMerger":FxAiVideoMerger,
	"FxAiVideoPreview":FxAiVideoPreview,
	"FxAiVideoManager":FxAiVideoManager,
    "FxAIGeneratorController": FxAIGeneratorController,
    "FxAiFrameGenerator": FxAiFrameGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FxAiAudioSegmenter": "凤希AI - 音频分段器",
    "FxAiAudioSegmenterV2": "凤希AI - 音频分段器V2",
    "FxAIAudioSegmentLoad": "凤希AI - 音频与帧数获取器",
    "FxAiSceneManager": "凤希AI - 视频场景管理 - 进Q群与更多的群友学习：775649071",
    "FxAiSceneManagerV2": "凤希AI - 视频场景管理V2 - 进Q群与更多的群友学习：775649071",
    "FxAiSceneLoad": "凤希AI - 场景数据加载器",
    "FxAiSceneLoadV2": "凤希AI - 场景数据加载器V2",
    "FxAiMultiLineText": "凤希AI - 场景生成器 - 进Q群与更多的群友学习：775649071",
    "FxAiMultiLineTextLoad": "凤希AI - 场景提示词加载器",
    "FxAiImageManager": "凤希AI - 图像管理器 - 群友扫地僧友情参与开发",
    "FxAiImageManagerV2": "凤希AI - 图像管理器V2",
    "FxAiLoadImageByIndex": "凤希AI - 图像管理器 - 图片加载",
	"FxAiPromptGenerator":"凤希AI - 提示词优化 - 本地Ollama",
	"FxAiSizeConfig":"凤希AI - 宽高设置",
	"FxAiMultiAudioLoad":"凤希AI - 多音频加载",
	"FxAiAudioManager":"凤希AI - 音频管理器",
	"FxAiLoadAudioByIndex":"凤希AI - 音频管理器 - 音频加载",
	"FxAiImageDownscale":"凤希AI - 图片缩小 - 按倍数",
	"FxAiPromptManager":"凤希AI - 提示词管理",
	"FxAiLoadPromptByIndex":"凤希AI - 提示词管理 - 提示词加载",
	"FxAiVideoGenerator":"凤希AI - 视频生成",
	"FxAiVideoMerger":"凤希AI - 视频合并",
	"FxAiVideoPreview":"凤希AI - 视频预览",
	"FxAiVideoManager":"凤希AI - 视频管理",
    "FxAIGeneratorController": "凤希AI - 生成控制器",
    "FxAiFrameGenerator": "凤希AI - 首尾帧生成器",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]