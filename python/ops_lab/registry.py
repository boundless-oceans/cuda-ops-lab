"""算子元数据注册表。

## 这个模块解决什么问题

测试和基准脚本都需要知道：*有哪些算子、每个算子有哪些变体、怎么调、跟谁比、
用什么形状测、搬运多少字节*。这些信息如果写在测试脚本里，每加一个算子就要改
中心文件，仓库很快会变成一团乱麻。

所以每个章节模块（`execution.py` / `elementwise.py` / …）自己声明元数据，
这里负责把它们汇总起来。**新增算子不需要改本文件。**

## 章节模块要声明什么

    NAME          章节名，如 "02_elementwise"。以章节号开头，天然可排序。
    VARIANT_NOTES 变体 -> 一句话"思路"，用于自动生成阶梯表
    OPS           算子描述符（见下）
    UTILITIES     可选。同样导出、但不是算子的辅助函数（一致性校验会跳过它们）

## 算子描述符

    OPS = {
        "<算子名>": {
            "variants":  {变体标签: 符号名 或 {"symbol":..., "args":[...]}},
            "reference": 参考实现，跑在 float64 上，签名与 kernel 的输入一致,
            "shapes":    [形状元组, ...]  必须含 0 / 1 / 非 2 的幂 / 跨 block 边界,
            "out_shape": 可选。shape -> 输出形状。省略时认为输出与输入同形状；
                         **归约这类输出比输入小的算子必须声明**，否则
                         `test_metadata.test_reference_runs` 会把正确的实现误判成错
            "gen":       可选。自定义输入生成器 gen(shape) -> tuple
            "bytes":     shape -> 搬运字节数（读 + 写）
            "flops":     shape -> 浮点运算次数
            "rtol"/"atol": 判据容差（默认 1e-5 / 1e-6）
            "torch":     可选。torch 对照实现，用于阶梯表的对照行
        }
    }

`variants` 支持两种写法，因为并不是所有算子都"一变体一符号"：

    "v1_naive": "elementwise_add_v1_naive"                 # 常规：一变体一符号
    "v1_smem0": {"symbol": "execution_bandwidth_probe",     # 一个符号 + 额外参数
                 "args": [1]}
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator

__all__ = [
    "VariantRef",
    "chapters",
    "iter_variants",
    "iter_ops",
    "call",
    "declared_symbols",
    "exported_symbols",
    "bound_arity",
    "consistency_report",
    "utility_symbols",
]

_chapters_cache: list[ModuleType] | None = None

# 不属于任何章节的基础设施符号。
#
# 它们同样会被导出，但既不是算子、也没有"变体"概念 —— 不测正确性、也不进阶梯表。
# 一致性校验把它们算作"已登记"，否则会被误报成"导出了却没登记"。
#
# 写在这里而不是某个章节模块里，是因为它不隶属于任何一章。
INFRASTRUCTURE_SYMBOLS = ("device_info",)


# ----------------------------------------------------------------- 章节发现


def chapters() -> list[ModuleType]:
    """发现所有声明了元数据的章节模块，按 NAME 排序。

    自动发现而不是维护一个手写清单 —— 新增一章只要建一个文件。
    模块名以 `_` 开头的跳过（`_build` / `_extension` 是基础设施，不是章节）。
    """
    global _chapters_cache
    if _chapters_cache is not None:
        return _chapters_cache

    # 注意：`__path__` 只在包的 __init__.py 里存在，子模块里没有。
    # 所以这里用文件位置推导包目录。
    package_dir = Path(__file__).resolve().parent

    found: list[ModuleType] = []
    for info in pkgutil.iter_modules([str(package_dir)]):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__package__}.{info.name}")
        if hasattr(module, "OPS") and hasattr(module, "NAME"):
            found.append(module)

    found.sort(key=lambda m: m.NAME)
    _chapters_cache = found
    return found


def _chapter_for(module: ModuleType) -> str:
    return str(module.NAME)


# ------------------------------------------------------------------- 变体


@dataclass(frozen=True)
class VariantRef:
    """指向扩展里一个具体符号的引用（含该变体需要的额外参数）。"""

    chapter: str
    op: str
    label: str
    symbol: str
    extra_args: tuple[Any, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        return f"{self.chapter}/{self.op}/{self.label}"


def _normalize_variant(value: Any) -> tuple[str, tuple[Any, ...]]:
    """把变体声明的两种写法统一成 (符号名, 额外参数)。"""
    if isinstance(value, str):
        return value, ()
    if isinstance(value, dict):
        symbol = value.get("symbol")
        if not isinstance(symbol, str):
            raise ValueError(f"变体声明缺少 symbol 字段：{value!r}")
        return symbol, tuple(value.get("args", ()))
    raise ValueError(f"无法识别的变体声明：{value!r}（应为字符串或 {{symbol, args}}）")


def iter_variants() -> Iterator[VariantRef]:
    """遍历所有 (章节, 算子, 变体)。"""
    for module in chapters():
        chapter = _chapter_for(module)
        for op_name, spec in module.OPS.items():
            for label, value in spec["variants"].items():
                symbol, extra = _normalize_variant(value)
                yield VariantRef(chapter, op_name, label, symbol, extra)


def iter_ops() -> Iterator[tuple[str, str, dict]]:
    """遍历所有 (章节名, 算子名, 描述符)。"""
    for module in chapters():
        chapter = _chapter_for(module)
        for op_name, spec in module.OPS.items():
            yield chapter, op_name, spec


# ------------------------------------------------------------------- 调用


def call(ref: VariantRef, *inputs: Any) -> Any:
    """按 VariantRef 调用扩展里的对应符号。

    变体自带的额外参数会追加在输入之后 —— 例如
    `execution_bandwidth_probe(x, 1)`。
    """
    from . import _extension  # 延迟导入：避免在没有构建产物时也去加载扩展

    fn = getattr(_extension.get(), ref.symbol)
    return fn(*inputs, *ref.extra_args)


def arity(spec: dict) -> int:
    """算子的输入个数，从参考实现的签名推断（避免多写一个字段、多一处不同步）。"""
    return len(inspect.signature(spec["reference"]).parameters)


# ------------------------------------------------------- 符号一致性校验


def declared_symbols() -> set[str]:
    """元数据里声明会用到的全部符号名。"""
    return {ref.symbol for ref in iter_variants()}


def utility_symbols() -> set[str]:
    """各章声明的辅助函数符号（导出但不是算子）。"""
    out: set[str] = set()
    for module in chapters():
        out.update(getattr(module, "UTILITIES", ()))
    return out


def exported_symbols() -> set[str]:
    """扩展实际导出的全部符号名。"""
    from . import _extension

    return {entry["name"] for entry in _extension.get().list_kernels()}


def bound_arity(symbol: str) -> int | None:
    """解析某个导出符号**实际接受**的参数个数。

    为什么不用 `inspect.signature`：它对 pybind11 的内建函数会抛
    `ValueError: no signature found for builtin`。但 pybind11 把签名放在了
    `__doc__` 的首行，形如：

        name(arg0: int, arg1: torch.Tensor) -> torch.Tensor

    解析不出来时返回 None —— 调用方应当**跳过**而不是当成 0，否则会误报。
    """
    import re

    from . import _extension

    fn = getattr(_extension.get(), symbol, None)
    if fn is None:
        return None

    first_line = (getattr(fn, "__doc__", None) or "").splitlines()
    if not first_line:
        return None

    match = re.search(r"\((.*)\)\s*->", first_line[0])
    if not match:
        return None

    inner = match.group(1).strip()
    if not inner:
        return 0
    return len([part for part in inner.split(",") if part.strip()])


def consistency_report() -> dict[str, list[str]]:
    """元数据声明的符号 ↔ 扩展导出的符号，双向对账。

    这是**无 GPU 环境下最有价值的检查**：它能挡住
      * 元数据里把符号名拼错（写了但导出里没有）
      * kernel 写了却没登记进元数据（导出了但没人用）
      * 忘了在 bind_XX.cpp 里导出
    这三类错误都不会有任何运行期征兆，只有对着名单才能发现。
    """
    declared = declared_symbols()
    exported = exported_symbols()
    accounted = declared | utility_symbols() | set(INFRASTRUCTURE_SYMBOLS)
    return {
        "declared_not_exported": sorted(declared - exported),
        "exported_not_declared": sorted(exported - accounted),
    }


# ------------------------------------------------------------------ 工具


def variant_notes(chapter_name: str) -> dict[str, str]:
    """某章的 变体 -> 思路 说明。"""
    for module in chapters():
        if _chapter_for(module) == chapter_name:
            return dict(getattr(module, "VARIANT_NOTES", {}))
    return {}


def chapter_module(chapter_name: str) -> ModuleType | None:
    for module in chapters():
        if _chapter_for(module) == chapter_name:
            return module
    return None


def find_variant(chapter: str, op: str, label: str) -> VariantRef | None:
    for ref in iter_variants():
        if ref.chapter == chapter and ref.op == op and ref.label == label:
            return ref
    return None


def all_ops() -> list[tuple[str, str, dict]]:
    return list(iter_ops())


def reference_of(spec: dict) -> Callable:
    return spec["reference"]
