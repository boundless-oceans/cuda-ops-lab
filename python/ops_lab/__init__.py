"""cuda-ops-lab 的 Python 包。

M1 / S1 阶段包含：

* ``envguard``     —— 剔除外来 `sys.path` 条目（保证任何 shell 下行为一致）
* ``build_config`` —— 构建参数探测，`check_env` 与 `_build` 的唯一事实来源
* ``_build``       —— 自研 ninja 构建
* ``_extension``   —— 扩展的加载与「缺失时自动构建」

典型用法::

    import ops_lab

    ext = ops_lab.ext()        # 必要时自动构建，然后返回扩展模块
    ext.list_kernels()         # 列出已登记的算子
    ext._probe()               # 查看编译期信息（ABI / 目标架构 / 编译器）

注意 ``ops_lab.ext`` 是**函数**不是模块；``ext()`` 返回的才是扩展模块。
"""

from . import build_config, envguard
from ._extension import extension_path, get as ext, is_built, list_kernels

__all__ = [
    "build_config",
    "envguard",
    "extension_path",
    "ext",
    "is_built",
    "list_kernels",
]
