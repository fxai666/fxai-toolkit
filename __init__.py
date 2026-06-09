# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权

import subprocess
import sys
import importlib
import os
from aiohttp import web
from server import PromptServer

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

required_packages = ["soundfile","psutil"]
for pkg in required_packages:
    install_package(pkg)

# ==============================================
# 【关键修复】把当前目录加入搜索路径，解决找不到模块问题
# ==============================================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ==============================================
# 【节点注册配置区】
# 格式：(节点类名, 模块文件名, 界面显示名称)
# ==============================================
NODE_REGISTRY = [
    # 音频类
    ("FxAiAudioSegmenter",        "fxai_audio_segments",        "凤希AI - 音频分段器"),
    ("FxAiAudioSegmenterV2",      "fxai_audio_segments_v2",     "凤希AI - 音频分段器V2"),
    ("FxAIAudioSegmentLoad",      "fxai_audio_frame_load",      "凤希AI - 音频与帧数获取器"),
    ("FxAiMultiAudioLoad",        "fxai_multi_audio_load",      "凤希AI - 多音频加载"),
    ("FxAiAudioManager",          "fxai_audio_manager",         "凤希AI - 音频管理器"),
    ("FxAiLoadAudioByIndex",      "fxai_audio_load",            "凤希AI - 音频管理器 - 音频加载"),
    ("FxAiSingleAudioProc",       "fxai_single_audio_proc",     "凤希AI - 单音频处理器"),
    ("FxAiAudioMerge",            "fxai_audio_merge",           "凤希AI - 多音频合并"),
    ("FxAiAudioAvgSplit",         "fxai_audio_avg_split",       "凤希AI - 音频平均分段器"),

    # 场景/文本类
    ("FxAiMultiLineText",         "fxai_multiline_text",        "凤希AI - 场景生成器 - 进Q群与更多的群友学习：775649071"),
    ("FxAiMultiLineTextLoad",     "fxai_multiline_text_load",   "凤希AI - 场景提示词加载器"),
    ("FxaiStoryBoard",            "fxai_story_board",           "凤希AI - 分镜生成器 - 进Q群与更多的群友学习：775649071"),
    ("FxaiStoryBoardV2",          "fxai_story_board",           "凤希AI - 分镜生成器V2 - 进Q群与更多的群友学习：775649071"),
    ("FxaiStoryBoardLoad",        "fxai_story_board_load",      "凤希AI - 分镜数据解析器"),
    ("FxAiSceneManager",          "fxai_scene_manager",         "凤希AI - 视频场景管理 - 进Q群与更多的群友学习：775649071"),
    ("FxAiSceneManagerV2",        "fxai_scene_manager_v2",      "凤希AI - 视频场景管理V2 - 进Q群与更多的群友学习：775649071"),
    ("FxAiSceneLoad",             "fxai_scene_load",            "凤希AI - 场景数据加载器"),
    ("FxAiSceneLoadV2",           "fxai_scene_load_v2",         "凤希AI - 场景数据加载器V2"),

    # 图像类
    ("FxAiImageManager",          "fxai_image_manager",         "凤希AI - 图像管理器 - 群友扫地僧友情参与开发"),
    ("FxAiImageManagerV2",        "fxai_image_manager_v2",      "凤希AI - 图像管理器V2"),
    ("FxAiLoadImageByIndex",      "fxai_image_load",            "凤希AI - 图像管理器 - 图片加载"),
    ("FxAiImageBatchLoad",        "fxai_image_batch_load",      "凤希AI - 图像管理器 - 批量获取"),
    ("FxAiImageGetSingle",        "fxai_image_get_single",      "凤希AI - 图像管理器 - 获取批量单图"),
    ("FxAiImageDownscale",        "fxai_resize_image_downscale","凤希AI - 图片缩小 - 按倍数"),
    ("FxAiImageBatchConcat",      "fxai_image_batch_concat",    "凤希AI - 图片列表合并"),
    ("FxAiImageLoopTile",         "fxai_image_loop_tile",       "凤希AI - 图片循环序列"),
	
    # 角色类
    ("FxAiCharacterAssets",       "fxai_character_assets",      "凤希AI - 角色资源管理器"),
    ("FxAiCharacterAssetsLoad",   "fxai_character_assets_load", "凤希AI - 角色资源数据解析器"),
    ("FxAiCharacterBatchLoad",    "fxai_character_batch_load",  "凤希AI - 角色加载器"),
    ("FxAiCharacterImageSelector","fxai_character_image_selector","凤希AI - 角色资源图片选择器"),

    # 提示词类
    ("FxAiPromptGenerator",       "fxai_prompt_optimization",   "凤希AI - 提示词优化 - 本地Ollama"),
    ("FxAiPromptManager",         "fxai_prompt_manager",        "凤希AI - 提示词管理"),
    ("FxAiLoadPromptByIndex",     "fxai_prompt_load",           "凤希AI - 提示词管理 - 提示词加载"),
    ("FxAiMultiPromptEditor",     "fxai_multi_prompt_editor",   "凤希AI - 分段场景 - 时间轴提示词管理器"),
    ("FxAiMultiPromptLoader",     "fxai_multi_prompt_loader",   "凤希AI - 分段场景 - 时间轴获取器"),
    ("FxAiPromptRelayEncode",     "fxai_prompt_relay_encode",   "凤希AI - 时序提示词编码器"),
    ("FxAiQwenEditEnhanced",      "fxai_qwen_edit_enhanced",    "凤希AI - 千问角色融合控制器"),
    ("FxAiQwenEditEnhancedV2",    "fxai_qwen_edit_enhanced_v2", "凤希AI - 千问多图编辑控制器"),

    # 视频类
    ("FxAiVideoGenerator",        "fxai_video_generator",       "凤希AI - 视频生成"),
    ("FxAiVideoGeneratorV2",      "fxai_video_generator_v2",    "凤希AI - 视频生成V2"),
    ("FxAiVideoGeneratorV3",      "fxai_video_generator_v3",    "凤希AI - 视频生成V3"),
    ("FxAiVideoMerger",           "fxai_video_merger",          "凤希AI - 视频合并"),
    ("FxAiVideoPreview",          "fxai_video_preview",         "凤希AI - 视频预览"),
    ("FxAiVideoManager",          "fxai_video_manager",         "凤希AI - 视频管理"),
    ("FxAiVideoLoad",             "fxai_video_load",            "凤希AI - 视频加载"),
    ("FxAiVideoToVR",             "fxai_video_tovr",            "凤希AI - 视频转VR"),

    # 控制/工具类
    ("FxAiSizeConfig",            "fxai_size_config",           "凤希AI - 宽高"),
    ("FxAiImageSizeConfig",       "fxai_image_size_config",     "凤希AI - 宽高-最大边长"),
    ("FxAiMaxRatioSize",          "fxai_max_ratio_size",        "凤希AI - 宽高-按比例"),
    ("FxAiIntToFloat",            "fxai_int_to_float",          "凤希AI - 整数转小数"),
    ("FxAIGeneratorController",   "fxai_generator_controller",  "凤希AI - 生成控制器"),
    ("FxAiFrameGenerator",        "fxai_frame_generator",       "凤希AI - 首尾帧生成器"),
    ("FxAiFrameGeneratorV2",      "fxai_frame_generator_v2",    "凤希AI - 首尾帧生成器V2"),
    ("FxAiStartEndIndex",         "fxai_start_end_index",       "凤希AI - 循环控制器"),
    ("FxAiAutoShutdown",          "fxai_auto_shut_down",        "凤希AI - 自动关机(GameOver)"),
    ("FxAiSegmentTotalFrames",    "fxai_segment_total_frames",  "凤希AI - 分段总帧数(向上取)"),
    ("FxAiTextPreview",           "fxai_text_preview",          "凤希AI - 文本预览"),
    ("FxAiImagePreview",          "fxai_image_preview",         "凤希AI - 图片预览"),
    ("FxAiReleaseResources",      "fxai_release_resources",     "凤希AI - 释放资源"),
    ("FxAiGetImagePath",          "fxai_get_image_path",        "凤希AI - 获取图片路径"),
    ("FxAiStrToNumber",           "fxai_str_to_number",         "凤希AI - 字符串转数字"),
    ("FxAiInputInt",              "fxai_input_int",             "凤希AI - 整数输入"),
    ("FxAiInputFloat",            "fxai_input_float",           "凤希AI - 小数输入"),
    ("FxAiFrameCalculator",       "fxai_frame_calculate",       "凤希AI - 帧数计算器"),

    # 潜空间类
    ("FxAiLatentClearReplace",    "fxai_latent_clear_replace",  "凤希AI - 潜空间清除与替换"),
    ("FxAiLatentGetFrames",       "fxai_latent_get_frames",     "凤希AI - 潜空间获取"),
    ("FxAiLatentGetFrameCount",   "fxai_latent_get_frame_count","凤希AI - 潜空间总数"),
    ("FxAiLTX23Sampler",          "fxai_ltx23_sampler",         "凤希AI - LTX2.3采集器"),
	
    # 模型/LoRA
    ("FxAiLoraLoader",            "fxai_lora_loader",           "凤希AI - LoRa加载器"),
    ("FxAiLtxvGuideFrames",       "fxai_ltxv_guide_frames",     "凤希AI - LTXV多帧引导器"),
	
    # 影视
    ("FxAiScreenManager",         "fxai_screen_manager",       "凤希AI - 影视场景管理 - 学习交流Q群：775649071"),
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for class_name, module_name, display_name in NODE_REGISTRY:
    try:
        print(f"✅[凤希AI] 正在加载 {display_name} 节点 - 进Q群与更多的群友学习：775649071")
        module = importlib.import_module(module_name)
        node_class = getattr(module, class_name)
        NODE_CLASS_MAPPINGS[class_name] = node_class
        NODE_DISPLAY_NAME_MAPPINGS[class_name] = display_name
    except Exception as e:
        print(f"❌[凤希AI] 节点加载失败: {display_name} → {str(e)}")

# Web 扩展
WEB_DIRECTORY = "./js"

# 导出
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# 加载 JS
js_file = os.path.join(os.path.dirname(__file__), "js/fxai_bottom_preview.js")
with open(js_file, "r", encoding="utf-8") as f:
    js_content = f.read()

@PromptServer.instance.routes.get("/fxai/image/bottom/preview.js")
async def fxai_bottom_preview(request):
    return web.Response(text=js_content, content_type="application/javascript")

try:
    PromptServer.instance.add_extra_js("/fxai/image/bottom/preview.js")
except:
    try:
        PromptServer.instance.add_extra_code('<script src="/fxai/image/bottom/preview.js"></script>')
    except:
        pass
