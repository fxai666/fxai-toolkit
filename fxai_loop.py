# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权

"""
凤希AI 循环 / 逻辑节点
======================

循环能力由本包自己实现，原先依赖的 `comfyui-easy-use` 已经移除，老工作流的
class_type 也已批量改成本文件里的节点名，所以不需要任何别名兼容层。

ComfyUI 的循环不是「跳回上一行」，而是节点在执行时返回
`{"result": ..., "expand": 子图}` —— ComfyUI 把这份子图挂到当前执行流上继续跑。
所以 for 循环的实质是：`结束节点` 每跑一次，就把「自己 + 循环体」重新展开一份，
把 index+1 传进去，直到 index 追上 total。展开的层数 = 循环次数，靠 ComfyUI
自己的执行队列递归，不占 Python 栈。

关键约定（参数布局照抄 easy-use，工作流是按位置填 `widgets_values` 的）：

- `flow` 用 `rawLink`，拿到的是**未解析的链接** `[node_id, slot]`，不是执行结果。
  结束节点靠 `flow[0]` 找到自己的开始节点，再从 dynprompt 里读它的 total。
- `initial_value0` 是内部状态位（当前 index），放在 `hidden` 里，前端看不到；
  `initial_value1..19` 才是用户能接的循环变量，20 个槽位与 easy-use 对齐。
- 循环次数为 0 时，开始节点把每个 value 输出换成 `ExecutionBlocker`，
  循环体（含里面的保存节点）整个不执行，结束节点把初值原样透传出去。

本文件里的节点：
    FxAiForLoopStart / FxAiForLoopEnd        计数循环
    FxAiWhileLoopStart / FxAiWhileLoopEnd    条件循环（for 循环内部用它递归）
    FxAiMathInt                              整数四则
    FxAiCompare                              比较
    FxAiIfElse                               条件选择（lazy，没选中的分支不求值）
    FxAiSimpleMath                           按公式算数
    FxAiString                               字符串常量
    FxAiImageScaleDown / FxAiImageConcat     缩放 / 拼接
    FxAiLoraNames                            LoRA 名选择
    FxAiCleanGpu / FxAiClearCache            清理显存 / 缓存
"""

import gc
import math

import folder_paths
import torch
from PIL import Image

from comfy_execution.graph_utils import ExecutionBlocker, GraphBuilder, is_link

MAX_FLOW_NUM = 20
ANY = "*"

# 结束类节点的 class_type：递归展开时要把它们排除在「循环体」之外，
# 否则每层都会把上一层结束节点当成循环体的一部分再展开一遍。
_LOOP_END_TYPES = {
    "FxAiForLoopEnd", "FxAiWhileLoopEnd",
    "easy forLoopEnd", "easy whileLoopEnd",
}


# ---------------------------------------------------------------------------
# 计数循环
# ---------------------------------------------------------------------------
class FxAiForLoopStart:
    """循环开始。输出 flow 给结束节点认，index 是当前第几次（从 0 数）。"""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "total": ("INT", {
                    "default": 1, "min": 0, "max": 100000, "step": 1,
                    "tooltip": "循环次数。0 = 一次都不跑，循环体（含里面的保存节点）全部不执行，"
                               "结束节点直接把下面的初值原样输出。",
                }),
            },
            "optional": {
                f"initial_value{i}": (ANY, {}) for i in range(1, MAX_FLOW_NUM)
            },
            "hidden": {
                # index 是内部状态，递归时由结束节点回填，不给前端显示
                "initial_value0": (ANY,),
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }
        return inputs

    RETURN_TYPES = ("FLOW_CONTROL", "INT") + (ANY,) * (MAX_FLOW_NUM - 1)
    RETURN_NAMES = ("flow", "index") + tuple(f"value{i}" for i in range(1, MAX_FLOW_NUM))
    FUNCTION = "for_loop_start"
    CATEGORY = "凤希AI/循环"

    def for_loop_start(self, total, prompt=None, extra_pnginfo=None, unique_id=None, **kwargs):
        i = kwargs.get("initial_value0", 0)

        if total <= 0:
            # 零次循环：把 value 输出全部 block 掉，循环体里的节点一个都不跑
            return tuple(["stub"] + [ExecutionBlocker(None)] * MAX_FLOW_NUM)

        graph = GraphBuilder()
        initial_values = {f"initial_value{n}": kwargs.get(f"initial_value{n}", None)
                          for n in range(1, MAX_FLOW_NUM)}
        # 计数循环就是「条件恒真的 while」：condition 传 total，靠结束节点递减
        graph.node("FxAiWhileLoopStart", condition=total, initial_value0=i, **initial_values)
        outputs = [kwargs.get(f"initial_value{n}", None) for n in range(1, MAX_FLOW_NUM)]
        return {
            "result": tuple(["stub", i] + outputs),
            "expand": graph.finalize(),
        }


class FxAiForLoopEnd:
    """循环结束。把 index+1 和本轮算出来的值喂回开始节点，直到 index 追上 total。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
            },
            "optional": {
                f"initial_value{i}": (ANY, {"rawLink": True})
                for i in range(1, MAX_FLOW_NUM)
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (ANY,) * (MAX_FLOW_NUM - 1)
    RETURN_NAMES = tuple(f"value{i}" for i in range(1, MAX_FLOW_NUM))
    FUNCTION = "for_loop_end"
    CATEGORY = "凤希AI/循环"

    def for_loop_end(self, flow, dynprompt=None, extra_pnginfo=None, unique_id=None, **kwargs):
        graph = GraphBuilder()
        start_id = flow[0]
        start_node = dynprompt.get_node(start_id)
        start_inputs = start_node["inputs"]

        if start_node["class_type"] not in ("FxAiForLoopStart", "easy forLoopStart"):
            raise RuntimeError(
                f"[凤希AI·循环] flow 接的不是循环开始节点（{start_node['class_type']}），"
                f"检查是不是把 flow 连错了"
            )

        total = start_inputs["total"]
        # 循环 0 次时用这些初值当结果
        skipped = {f"initial_value{i}": start_inputs.get(f"initial_value{i}", None)
                   for i in range(1, MAX_FLOW_NUM)}

        # total 是手填的常数：这里就能判断出「不跑」。
        # 直接返回 Blocker 而不是透传初值 —— 初值多半是 None（循环变量第一次用），
        # 透传下去下游会拿 None 报错；Blocker 让整条下游静默跳过。
        if not is_link(total) and total <= 0:
            return tuple(ExecutionBlocker(None) for _ in range(MAX_FLOW_NUM - 1))

        # index + 1 < total 就继续下一轮
        nxt = graph.node("FxAiMathInt", a=[start_id, 1], b=1, operation="add")
        cond = graph.node("FxAiCompare", a=nxt.out(0), b=total, comparison="a < b")
        values = {f"initial_value{i}": kwargs.get(f"initial_value{i}", None)
                  for i in range(1, MAX_FLOW_NUM)}
        close = graph.node("FxAiWhileLoopEnd", flow=flow, condition=cond.out(0),
                           initial_value0=nxt.out(0), **values)
        outputs = [close.out(i) for i in range(1, MAX_FLOW_NUM)]

        # total 是连线进来的：跑不跑要等执行到才知道，用 lazy 的 ifElse 二选一。
        # 没选中的那个分支（被 block 的循环体）根本不会被求值，所以不会报错。
        if is_link(total):
            runs = graph.node("FxAiCompare", a=total, b=0, comparison="a > b")
            for i in range(1, MAX_FLOW_NUM):
                fallback = skipped[f"initial_value{i}"]
                if fallback is None:
                    continue
                outputs[i - 1] = graph.node("FxAiIfElse", boolean=runs.out(0),
                                            on_true=close.out(i),
                                            on_false=fallback).out(0)

        return {
            "result": tuple(outputs),
            "expand": graph.finalize(),
        }


# ---------------------------------------------------------------------------
# 条件循环（for 循环的底座，也能单独用）
# ---------------------------------------------------------------------------
class FxAiWhileLoopStart:
    """条件循环开始。condition 为假时 value 输出全部 block，循环体不执行。"""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "condition": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "为真才跑这一轮。接结束节点的条件输出，或自己接一个比较节点。",
                }),
            },
            "optional": {},
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"][f"initial_value{i}"] = (ANY,)
        return inputs

    RETURN_TYPES = ("FLOW_CONTROL",) + (ANY,) * MAX_FLOW_NUM
    RETURN_NAMES = ("flow",) + tuple(f"value{i}" for i in range(MAX_FLOW_NUM))
    FUNCTION = "while_loop_open"
    CATEGORY = "凤希AI/循环"

    def while_loop_open(self, condition, **kwargs):
        values = []
        for i in range(MAX_FLOW_NUM):
            v = kwargs.get(f"initial_value{i}", None)
            values.append(v if condition else ExecutionBlocker(None))
        return tuple(["stub"] + values)


class FxAiWhileLoopEnd:
    """条件循环结束。condition 为真就把「自己 + 循环体」再展开一份递归下去。"""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
                "condition": ("BOOLEAN", {}),
            },
            "optional": {},
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"][f"initial_value{i}"] = (ANY,)
        return inputs

    RETURN_TYPES = (ANY,) * MAX_FLOW_NUM
    RETURN_NAMES = tuple(f"value{i}" for i in range(MAX_FLOW_NUM))
    FUNCTION = "while_loop_close"
    CATEGORY = "凤希AI/循环"

    @staticmethod
    def _walk_upstream(node_id, dynprompt, upstream, parent_ids):
        """顺着输入往上走，收集「循环体」里有哪些节点。"""
        node = dynprompt.get_node(node_id)
        if "inputs" not in node:
            return
        for value in node["inputs"].values():
            if not is_link(value):
                continue
            parent_id = value[0]
            display_id = dynprompt.get_display_node_id(parent_id)
            class_type = dynprompt.get_node(display_id)["class_type"]
            if class_type not in _LOOP_END_TYPES:
                parent_ids.append(display_id)
            if parent_id not in upstream:
                upstream[parent_id] = []
                FxAiWhileLoopEnd._walk_upstream(parent_id, dynprompt, upstream, parent_ids)
            upstream[parent_id].append(node_id)

    @staticmethod
    def _attach_output_nodes(dynprompt, upstream, output_nodes, parent_ids):
        """循环体里的输出节点（SaveImage 等）也要算进 contained，否则整轮不保存。"""
        for parent_id in upstream:
            display_id = dynprompt.get_display_node_id(parent_id)
            for output_id, link in output_nodes.items():
                if link[0] in parent_ids and display_id == link[0] \
                        and output_id not in upstream[parent_id]:
                    if "." in parent_id:
                        parts = parent_id.split(".")
                        parts[-1] = output_id
                        upstream[parent_id].append(".".join(parts))
                    else:
                        upstream[parent_id].append(output_id)

    def _collect(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child in upstream[node_id]:
            if child not in contained:
                contained[child] = True
                self._collect(child, upstream, contained)

    def while_loop_close(self, flow, condition, dynprompt=None, extra_pnginfo=None,
                         unique_id=None, **kwargs):
        if not condition:
            return tuple(kwargs.get(f"initial_value{i}", None) for i in range(MAX_FLOW_NUM))

        this_node = dynprompt.get_node(unique_id)
        upstream = {}
        parent_ids = []
        self._walk_upstream(unique_id, dynprompt, upstream, parent_ids)
        parent_ids = list(set(parent_ids))

        prompts = dynprompt.get_original_prompt()
        from nodes import NODE_CLASS_MAPPINGS as ALL
        output_nodes = {}
        for nid, node in prompts.items():
            if "inputs" not in node:
                continue
            cls = ALL.get(node["class_type"])
            if cls is None or not getattr(cls, "OUTPUT_NODE", False):
                continue
            for v in node["inputs"].values():
                if is_link(v):
                    output_nodes[nid] = v

        graph = GraphBuilder()
        self._attach_output_nodes(dynprompt, upstream, output_nodes, parent_ids)
        contained = {}
        start_id = flow[0]
        self._collect(start_id, upstream, contained)
        contained[unique_id] = True
        contained[start_id] = True

        # 把循环体整块复制进子图：结束节点自己换成 Recurse，其余保持原 id
        for nid in contained:
            original = dynprompt.get_node(nid)
            graph.node(original["class_type"], "Recurse" if nid == unique_id else nid)
        for nid in contained:
            original = dynprompt.get_node(nid)
            node = graph.lookup_node("Recurse" if nid == unique_id else nid)
            node.set_override_display_id(nid)
            for key, value in original["inputs"].items():
                if is_link(value) and value[0] in contained:
                    node.set_input(key, graph.lookup_node(value[0]).out(value[1]))
                else:
                    node.set_input(key, value)

        new_start = graph.lookup_node(start_id)
        for i in range(MAX_FLOW_NUM):
            key = f"initial_value{i}"
            new_start.set_input(key, kwargs.get(key, None))

        recurse = graph.lookup_node("Recurse")
        return {
            "result": tuple(recurse.out(i) for i in range(MAX_FLOW_NUM)),
            "expand": graph.finalize(),
        }


# ---------------------------------------------------------------------------
# 逻辑
# ---------------------------------------------------------------------------
_MATH_INT_OPS = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a // b,
    "modulo": lambda a, b: a % b,
    "power": lambda a, b: a ** b,
}


class FxAiMathInt:
    """整数运算。循环里最常见的用法：index + 1。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("INT", {"default": 0, "min": -0xFFFFFFFFFFFFFFFF,
                              "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "b": ("INT", {"default": 0, "min": -0xFFFFFFFFFFFFFFFF,
                              "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "operation": (list(_MATH_INT_OPS), {
                    "default": "add",
                    "tooltip": "divide 是整除（向下取整），跟 Python // 一致。",
                }),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("INT",)
    FUNCTION = "calc"
    CATEGORY = "凤希AI/逻辑"

    def calc(self, a, b, operation):
        if operation == "divide" and b == 0:
            raise ValueError("[凤希AI·整数运算] 除数为 0，请改一个非零的 b")
        return (_MATH_INT_OPS[operation](a, b),)


_COMPARE_OPS = {
    "a == b": lambda a, b: a == b,
    "a != b": lambda a, b: a != b,
    "a < b": lambda a, b: a < b,
    "a > b": lambda a, b: a > b,
    "a <= b": lambda a, b: a <= b,
    "a >= b": lambda a, b: a >= b,
    "a > 0": lambda a, b: a > 0,
    "a <= 0": lambda a, b: a <= 0,
    "b > 0": lambda a, b: b > 0,
    "b <= 0": lambda a, b: b <= 0,
}


class FxAiCompare:
    """比较两个值，输出 BOOLEAN。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": (ANY, {}),
                "b": (ANY, {}),
                "comparison": (list(_COMPARE_OPS), {
                    "default": "a == b",
                    "tooltip": "后四项只看一个操作数（`a > 0` 这类），另一个接不接都行。",
                }),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "compare"
    CATEGORY = "凤希AI/逻辑"

    def compare(self, a=0, b=0, comparison="a == b"):
        return (_COMPARE_OPS[comparison](a, b),)


class FxAiIfElse:
    """条件选择。没选中的那个分支是 lazy 的，不会被求值。

    这一点决定了「循环 0 次」能否优雅处理：被 block 的循环体挂在 on_true 上，
    boolean 为假时它根本不会被拉起来执行。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boolean": ("BOOLEAN", {"default": True}),
                "on_true": (ANY, {"lazy": True}),
                "on_false": (ANY, {"lazy": True}),
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("*",)
    FUNCTION = "pick"
    CATEGORY = "凤希AI/逻辑"

    @classmethod
    def check_lazy_status(cls, boolean=True, on_true=None, on_false=None):
        if boolean and on_true is None:
            return ["on_true"]
        if not boolean and on_false is None:
            return ["on_false"]
        return []

    def pick(self, boolean=True, on_true=None, on_false=None):
        return (on_true if boolean else on_false,)


# ---------------------------------------------------------------------------
# 清理
# ---------------------------------------------------------------------------
# 释放显存 / 内存的实现在 fxai_release_resources 里，本文件只做「接一个值、
# 清一遍、原样透传」的壳，清理逻辑不另写一份。
from fxai_release_resources import release_resources


class _CleanPassThrough:
    """清理节点的共用骨架：拿一个值、清一遍、原样透传。子类只实现 clean()。

    anything 不是拿来用的，是用来定执行顺序的 —— 接了上一环的输出，
    这个节点才会等上游跑完再清。
    """

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("output",)
    FUNCTION = "clean"
    CATEGORY = "凤希AI/清理"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": (ANY, {"tooltip": "接上一环的输出，本节点只负责清理，不动数据。"}),
            },
        }

    def clean(self, anything):
        raise NotImplementedError


class FxAiCleanGpu(_CleanPassThrough):
    """清显存：卸载所有模型 + 清 ComfyUI 的显存缓存，值原样透传。"""

    def clean(self, anything):
        release_resources(unload_models=True, empty_cuda=True)
        return (anything,)


class FxAiClearCache(_CleanPassThrough):
    """清缓存：只做一次垃圾回收，值原样透传。

    不碰 ComfyUI 的执行缓存，也不卸模型 —— 前者会让下游整条重跑（不是省内存，
    是白干一遍），后者会把下一轮要用的权重重新搬回显存。
    """

    def clean(self, anything):
        gc.collect()
        return (anything,)


# ---------------------------------------------------------------------------
# 类型透传
# ---------------------------------------------------------------------------
class FxAiString:
    """字符串常量。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("STRING", {"default": "", "multiline": False})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "passthrough"
    CATEGORY = "凤希AI/逻辑"

    def passthrough(self, value):
        return (value,)


# ---------------------------------------------------------------------------
# 公式计算
# ---------------------------------------------------------------------------
_MATH_NAMESPACE = {
    "abs": abs, "pow": pow, "round": round, "min": min, "max": max,
    "ceil": math.ceil, "floor": math.floor, "sqrt": math.sqrt,
    "exp": math.exp, "log": math.log, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "pi": math.pi, "e": math.e,
}


class FxAiSimpleMath:
    """按公式算一个数。公式里用 a / b / c 引用三个输入。

    支持 + - * / // % ** 与比较运算，函数见 _MATH_NAMESPACE。
    例：`(b*c)/a+0.8`、`ceil(a/b)`、`round(a*b, 2)`。

    同时给出 int / float / boolean 三个输出，接哪个都行 —— 布尔是「结果非 0」。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "数学公式，用 a / b / c 引用下面的三个输入。\n"
                               "例：(b*c)/a+0.8、ceil(a/b)、round(a*b, 2)",
                }),
            },
            "optional": {
                "a": (ANY, {}),
                "b": (ANY, {}),
                "c": (ANY, {}),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("int", "float", "boolean")
    FUNCTION = "calc"
    CATEGORY = "凤希AI/逻辑"

    def calc(self, value, a=0, b=0, c=0):
        try:
            result = eval(str(value), {"__builtins__": {}},
                          {**_MATH_NAMESPACE, "a": a, "b": b, "c": c})
            return (int(result), float(result), int(result) != 0)
        except Exception as exc:
            print(f"⚠️[凤希AI·公式] 算不出来（{value}）：{exc}")
            return (0, 0.0, False)


# ---------------------------------------------------------------------------
# 图像
# ---------------------------------------------------------------------------
# 张量 ↔ PIL 用包里 fxai_image_utils 的共用实现，本文件直接复用。
from fxai_image_utils import pil_to_tensor, tensor_to_pil


class FxAiImageScaleDown:
    """把整批图缩放到指定宽高。

    crop=center 时先按目标比例居中裁掉多余部分再缩，不会压扁；
    crop=disabled 直接缩放，比例会变。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
                "crop": (["disabled", "center"], {
                    "default": "disabled",
                    "tooltip": "center = 先按目标比例居中裁切再缩放，不改变画面比例。",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "scale_down"
    CATEGORY = "凤希AI/图像"

    def scale_down(self, images, width, height, crop):
        src = images
        if crop == "center":
            old_w, old_h = images.shape[2], images.shape[1]
            old_aspect, new_aspect = old_w / old_h, width / height
            x = y = 0
            if old_aspect > new_aspect:
                x = round((old_w - old_w * (new_aspect / old_aspect)) / 2)
            elif old_aspect < new_aspect:
                y = round((old_h - old_h * (old_aspect / new_aspect)) / 2)
            src = images[:, y:old_h - y, x:old_w - x, :]

        out = [pil_to_tensor(tensor_to_pil(im).convert("RGB").resize((width, height), Image.LANCZOS))
               for im in src]
        return (torch.cat(out, dim=0),)


class FxAiImageConcat:
    """两张图拼一起。方向 right/down/left/up。

    match_image_size=True 时把 image2 等比缩到与 image1 同高（左右拼）或同宽（上下拼），
    否则两张尺寸不一致会直接报错。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "direction": (["right", "down", "left", "up"], {"default": "right"}),
                "match_image_size": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "两张图尺寸不同时必须开：把 image2 等比缩到与 image1 同高/同宽后再拼。",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "concat"
    CATEGORY = "凤希AI/图像"

    def concat(self, image1, image2, direction, match_image_size):
        if image1 is None:
            return (image2,)
        if image2 is None:
            return (image1,)

        horizontal = direction in ("right", "left")
        if match_image_size:
            # 逐张按比例缩到与 image1 同高（左右拼）或同宽（上下拼），不补边也不裁 ——
            # 比例本身没变，等比缩完尺寸就刚好吻合，补边反而会多出黑条。
            target_h = image1.shape[1]
            target_w = image1.shape[2]
            scaled = []
            for im in image2:
                pil = tensor_to_pil(im).convert("RGB")
                if horizontal:
                    size = (max(1, round(pil.width / pil.height * target_h)), target_h)
                else:
                    size = (target_w, max(1, round(pil.height / pil.width * target_w)))
                scaled.append(pil_to_tensor(pil.resize(size, Image.LANCZOS)))
            image2 = torch.cat(scaled, dim=0)

        # 批大小对齐：少的那批循环补齐，免得 cat 直接炸
        b1, b2 = image1.shape[0], image2.shape[0]
        if b1 != b2:
            n = max(b1, b2)
            if b1 != n:
                image1 = image1.repeat((n + b1 - 1) // b1, 1, 1, 1)[:n]
            if b2 != n:
                image2 = image2.repeat((n + b2 - 1) // b2, 1, 1, 1)[:n]

        if horizontal and image1.shape[1] != image2.shape[1]:
            raise ValueError(
                f"[凤希AI·图片拼接] 两张图高度不一致（{image1.shape[1]} vs {image2.shape[1]}），"
                "请把「匹配图片尺寸」打开")
        if not horizontal and image1.shape[2] != image2.shape[2]:
            raise ValueError(
                f"[凤希AI·图片拼接] 两张图宽度不一致（{image1.shape[2]} vs {image2.shape[2]}），"
                "请把「匹配图片尺寸」打开")

        if direction == "right":
            row = torch.cat((image1, image2), dim=2)
        elif direction == "down":
            row = torch.cat((image1, image2), dim=1)
        elif direction == "left":
            row = torch.cat((image2, image1), dim=2)
        else:
            row = torch.cat((image2, image1), dim=1)
        return (row,)


# ---------------------------------------------------------------------------
# 资源名
# ---------------------------------------------------------------------------
class FxAiLoraNames:
    """LoRA 文件名选择器。输出字符串，接给 LoRA 加载节点的名字输入。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"lora_name": (folder_paths.get_filename_list("loras"), {})}}

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("lora_name",)
    FUNCTION = "pick"
    CATEGORY = "凤希AI/资源"

    def pick(self, lora_name):
        return (lora_name,)
