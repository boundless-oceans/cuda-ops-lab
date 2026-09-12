"""加载（必要时先构建）CUDA 扩展 ops_lab_ext。

## 为什么不用 `import ops_lab_ext`

扩展产物是 `build/ext/ops_lab_ext.so`：

* 文件名没有 Python 的 ABI tag（`cpython-39-x86_64-linux-gnu`），
  不符合标准扩展模块的命名约定；
* 位置由构建脚本决定，并且不在 `sys.path` 上。

所以按路径用 importlib 加载，而不是 `import`。

## 何时构建

`get()` 发现 `.so` 不存在时会自动构建一次 —— 第一次用就自动编译，很方便；
代价是**首次调用会慢几十秒**。想要明确的控制就用命令行的
`python python/ops_lab/_build.py`，或调用 `get(rebuild=True)` 强制重编。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_module: ModuleType | None = None


def _build_module():
    """延迟导入 `_build`。

    不能在模块顶层导入：`_build` 在顶层执行 ``from ops_lab import build_config``，
    而 ``ops_lab/__init__.py`` 会导入本模块 —— 顶层导入会形成循环，
    并且一旦 `__init__` 里的导入顺序被调整就会炸。
    """
    from . import _build

    return _build


def extension_path() -> Path:
    """扩展 `.so` 的路径（不保证文件已存在）。"""
    return _build_module().extension_path()


def is_built() -> bool:
    """扩展是否已经构建过。"""
    return extension_path().is_file()


def get(rebuild: bool = False) -> ModuleType:
    """返回已加载的扩展模块；缺失时自动构建。

    参数
        rebuild: True 时清空 `build/` 重新编译（改了编译参数后用）。
    """
    global _module
    if _module is not None and not rebuild:
        return _module

    build_mod = _build_module()
    path = build_mod.extension_path()

    if rebuild or not path.is_file():
        reason = "需要重新构建" if rebuild else "尚未构建"
        print(f"[ops_lab] 扩展{reason}：{path}")
        if build_mod.build(clean=rebuild) != 0:
            raise RuntimeError(
                "扩展构建失败。请手动运行以查看完整输出：\n"
                "    python python/ops_lab/_build.py -v"
            )

    _module = _load(path)
    return _module


def list_kernels() -> list[dict]:
    """列出所有已登记的导出算子（name/chapter/variant/description）。"""
    return get().list_kernels()


def _load(path: Path) -> ModuleType:
    name = _build_module().EXTENSION_NAME

    cached = sys.modules.get(name)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法为 {path} 创建 import spec")

    module = importlib.util.module_from_spec(spec)
    # 先注册进 sys.modules 再执行：模块内的自引用与 pybind11 的某些类型转换
    # 依赖它。失败时要把半成品摘掉，否则下次 import 会拿到坏模块。
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
