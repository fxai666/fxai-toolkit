import os
import csv
import json
import random
import folder_paths

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PLUGIN_DIR, "data", "prompt_library")

def load_csv_options(file_path):
    """读取 CSV，返回选项列表，每个选项格式为 "en|zh" 字符串，并添加空选项"""
    options = [""]
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                en = row.get("en", "").strip()
                zh = row.get("zh", "").strip()
                if en:
                    options.append(f"{en}|{zh}")
    except Exception as e:
        print(f"[CF_PromptLibrary] 加载 {file_path} 失败: {e}")
    return options

def load_categories():
    cat_file = os.path.join(DATA_DIR, "categories.json")
    if not os.path.exists(cat_file):
        print(f"[CF_PromptLibrary] 未找到 categories.json: {cat_file}")
        return []
    with open(cat_file, "r", encoding="utf-8") as f:
        return json.load(f)

class CF_PromptLibrary:
    CATEGORIES = load_categories()

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "output_language": (["en", "zh"], {"default": "en"}),
                "random_all": ("BOOLEAN", {"default": False}),
            }
        }
        # 动态添加每个分类的 COMBO 控件
        for cat in cls.CATEGORIES:
            csv_path = os.path.join(DATA_DIR, cat["file"])
            if os.path.exists(csv_path):
                options = load_csv_options(csv_path)
            else:
                options = [""]
            inputs["required"][cat["id"]] = (options, {"default": ""})
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate"
    CATEGORY = "CF工具包"   # 注意：必须与其他节点一致，显示在 CF工具包 分组下

    def generate(self, output_language, random_all, **kwargs):
        if not self.CATEGORIES:
            return ("请先配置 data/prompt_library/categories.json",)
        
        selected_items = {}
        for cat in self.CATEGORIES:
            cat_id = cat["id"]
            if random_all:
                # 随机模式：忽略 kwargs，从该分类的非空选项中随机选一个
                csv_path = os.path.join(DATA_DIR, cat["file"])
                if os.path.exists(csv_path):
                    options = load_csv_options(csv_path)
                    non_empty = [opt for opt in options if opt]
                    if non_empty:
                        selected_items[cat_id] = random.choice(non_empty)
                    else:
                        selected_items[cat_id] = ""
                else:
                    selected_items[cat_id] = ""
            else:
                selected_items[cat_id] = kwargs.get(cat_id, "")
        
        prompt_parts = []
        for cat in self.CATEGORIES:
            selected = selected_items.get(cat["id"], "")
            if selected:
                parts = selected.split("|")
                if len(parts) == 2:
                    en, zh = parts
                    if output_language == "zh":
                        text = zh
                    else:
                        text = en
                    if text:
                        prompt_parts.append(text)
        prompt = ", ".join(prompt_parts)
        return (prompt,)

NODE_CLASS_MAPPINGS = {"CF_PromptLibrary": CF_PromptLibrary}
NODE_DISPLAY_NAME_MAPPINGS = {"CF_PromptLibrary": "CF 提示词库"}