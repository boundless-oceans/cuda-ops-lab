"""cuda-ops-lab 的 Python 包。

## 模块分工

* ``envguard``     —— 剔除外来 `sys.path` 条目（保证任何 shell 下行为一致）
* ``build_config`` —— 构建参数探测，`check_env` 与 `_build` 的唯一事实来源
* ``_build``       —— 自研 ninja 构建
* ``_extension``   —— 扩展的加载与「缺失时自动构建」
* ``registry``     —— **算子元数据注册表**：测试与基准的公共入口
* ``execution`` / ``elementwise`` / … —— 各章的元数据（自动发现，新增章节只加文件）

## 典型用法

加载扩展::

    import ops_lab

    ext = ops_lab.ext()        # 必要时自动构建，然后返回扩展模块
    ext.list_kernels()         # 列出已登记的算子
    ext._probe()               # 查看编译期信息（ABI / 目标架构 / 编译器）

遍历算子元数据::

    from ops_lab import registry

    for ref in registry.iter_variants():
        print(ref, "->", ref.symbol)

    # 元数据声明的符号 ↔ 扩展实际导出，双向对账
    print(registry.consistency_report())

注意 `ops_lab.ext` 是**函数**不是模块；`ext()` 返回的才是扩展模块。
"""

from . import build_config, envguard, registry
from ._extension import extension_path, get as ext, is_built, list_kernels

__all__ = [
    "build_config",
    "envguard",
    "registry",
    "extension_path",
    "ext",
    "is_built",
    "list_kernels",
]
