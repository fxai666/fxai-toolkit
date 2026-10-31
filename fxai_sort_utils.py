# Copyright (c) 2026 凤希AI/www.fxai.site
# Licensed under MIT License
# 商用需购买商业授权
#
# 文件列表排序公共层：全包统一使用「自然排序」。
#
# 为什么必须有这个模块：
#   os.listdir() 的返回顺序完全取决于文件系统——Windows(NTFS) 近似按文件名，
#   Linux(ext4/xfs) 是目录哈希序。所以同一份 000/001/002 序列的文件，
#   在 Windows 上看着是对的，在 Linux 上就是乱的。凡是给用户看列表的地方
#   都必须显式排序，不能依赖 listdir 的原始顺序。

import re

_NUM_RE = re.compile(r'(\d+)')


def natural_key(name):
    """自然排序键。

    - 数字段按数值比较：2.png 排在 10.png 之前（纯字符串排序会得到 10 < 2）
    - 文字段忽略大小写
    - 末尾用原名兜底，保证键元素类型一致、可直接比较，且排序结果稳定

    000.png / 001.png / 002.png 这类零填充序列同样正确。
    """
    text = str(name)
    parts = _NUM_RE.split(text.lower())
    key = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:  # 捕获组：纯数字段
            key.append((1, int(part), ""))
        else:             # 文字段
            key.append((0, 0, part))
    key.append((0, 0, text))  # 原名兜底，避免键完全相同导致顺序不稳
    return key


def sort_names(names):
    """对文件名列表原地自然排序，并返回该列表，方便链式调用。"""
    names.sort(key=natural_key)
    return names
