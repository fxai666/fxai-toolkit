import os

# 支持的提示词文件格式（仅txt，可扩展）
PROMPT_EXTENSIONS = ('.txt',)

class FxAiLoadPromptByIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词文件夹路径": ("STRING", {"multiline": False}),
                "提示词索引": ("INT", {"default": 0, "min": 0}),
            },
            "optional": {
                "刷新标记": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("提示词内容", "当前提示词路径", "总提示词数量")
    FUNCTION = "load_prompt"
    CATEGORY = "凤希AI"

    def load_prompt(self, 提示词文件夹路径, 提示词索引, 刷新标记=0):
        # 1. 清理路径，去除首尾空格
        folder_path = 提示词文件夹路径.strip()
        
        # 2. 检查文件夹是否存在
        if not os.path.isdir(folder_path):
            raise RuntimeError(f"提示词文件夹不存在：{folder_path}")
        
        # 3. 获取文件夹里所有提示词文件（按文件名排序）
        prompt_files = []
        for f in os.listdir(folder_path):
            if f.lower().endswith(PROMPT_EXTENSIONS):
                full_path = os.path.join(folder_path, f)
                prompt_files.append(full_path)
        
        # 按文件名排序（保证每次加载顺序一致）
        prompt_files.sort()
        
        # 4. 检查是否有提示词文件
        total_prompts = len(prompt_files)
        if total_prompts == 0:
            raise RuntimeError(f"提示词文件夹中未找到有效文件：{folder_path}")
        
        # 5. 检查索引是否越界
        if 提示词索引 >= total_prompts:
            raise RuntimeError(f"提示词索引越界！共 {total_prompts} 个提示词文件，索引范围：0 ~ {total_prompts-1}")
        
        # 6. 加载选中的提示词文件内容
        target_path = prompt_files[提示词索引]
        try:
            # 读取文件（UTF-8编码，兼容中文；若有GBK文件可扩展编码判断）
            with open(target_path, 'r', encoding='utf-8') as f:
                prompt_content = f.read().strip()
        except Exception as e:
            raise RuntimeError(f"读取提示词文件失败 {target_path}：{str(e)}")
        
        # 返回：提示词内容、文件路径、总数量
        return (prompt_content, target_path, total_prompts)