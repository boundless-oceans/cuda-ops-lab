"""零依赖测试收集器。

## 为什么自己写

环境里没有 pytest（见 `environment.yml`），而测试需求很小：注册用例、跑、报告。
为这点需求引入一个依赖不划算，何况自研还能顺手实现下面这条关键行为。

## 关键行为：无 GPU 时**跳过**而不是失败

每个用例可以标记 `gpu=True`。没有可见 GPU 时这类用例会被跳过并在报告里单列。
这样同一套测试在两种机器上都是"全绿"：
  * 没有显卡的机器/CI —— 跑完离线部分（元数据一致性等）
  * 你的开发机 —— 离线部分 + 全部数值正确性

如果无 GPU 时直接判失败，离线检查就会被淹没在噪声里，等于没人看。

## 约定

* 用 `@test(...)` 装饰器注册
* 测试函数**正常返回**表示通过；抛异常表示失败，异常的字符串就是失败原因
* 失败信息要写清"期望什么、实际什么、在哪一个位置"，否则排查时还得重现一遍
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

__all__ = ["test", "run", "report", "has_gpu", "assert_close", "assert_true", "Result"]

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


@dataclass
class Case:
    name: str
    group: str
    fn: Callable[[], None]
    gpu: bool = False


@dataclass
class Result:
    case: Case
    status: str
    detail: str = ""


_CASES: list[Case] = []


def test(name: str | None = None, group: str = "default", gpu: bool = False):
    """把一个函数注册为测试用例。

    参数
        name:  用例名；省略时用函数名
        group: 分组，通常用章节名（如 "02_elementwise"），供 --chapters 过滤
        gpu:   True 表示需要 GPU，无 GPU 时跳过而不是失败
    """

    def decorator(fn: Callable[[], None]) -> Callable[[], None]:
        _CASES.append(Case(name=name or fn.__name__, group=group, fn=fn, gpu=gpu))
        return fn

    return decorator


def cases() -> list[Case]:
    return list(_CASES)


# ------------------------------------------------------------------ GPU 检测


_gpu_cache: bool | None = None


def has_gpu() -> bool:
    global _gpu_cache
    if _gpu_cache is None:
        try:
            import torch

            _gpu_cache = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001 - 探测失败一律当作没有 GPU
            _gpu_cache = False
    return _gpu_cache


# -------------------------------------------------------------------- 断言


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def assert_close(got, expected, rtol: float, atol: float, what: str = "") -> None:
    """比较 kernel 输出与 float64 参考实现。

    rtol == atol == 0 时要求**精确相等**（整数索引、纯拷贝这类算子用得上）。
    否则判据是 `|got - expected| <= atol + rtol * |expected|`。
    """
    import torch

    g = got.detach().to("cpu")
    e = expected.detach().to("cpu")

    if tuple(g.shape) != tuple(e.shape):
        raise AssertionError(f"{what}形状不符：实际 {tuple(g.shape)}，期望 {tuple(e.shape)}")

    gd = g.double()
    ed = e.double()

    if rtol == 0.0 and atol == 0.0:
        if not torch.equal(gd, ed):
            bad = (gd != ed).nonzero().flatten()
            first = bad[:1].tolist()
            raise AssertionError(
                f"{what}要求精确相等，但有 {bad.numel()} / {gd.numel()} 个位置不同"
                f"（首个下标 {first}：实际 {float(gd.flatten()[bad[0]]):.6g}，"
                f"期望 {float(ed.flatten()[bad[0]]):.6g}）"
            )
        return

    diff = (gd - ed).abs()
    tol = atol + rtol * ed.abs()
    exceeded = diff > tol
    if bool(exceeded.any()):
        n = int(exceeded.sum())
        idx = int(exceeded.flatten().nonzero()[0])
        raise AssertionError(
            f"{what}超出容差（atol={atol:g}, rtol={rtol:g}）："
            f"{n} / {gd.numel()} 个位置不合格；"
            f"最大偏差 {float(diff.max()):.3e}（下标 {idx}）；"
            f"该点 实际 {float(gd.flatten()[idx]):.9g}，期望 {float(ed.flatten()[idx]):.9g}"
        )


# -------------------------------------------------------------------- 执行


def run(chapters: Iterable[str] | None = None, verbose: bool = False) -> list[Result]:
    """跑所有（或按章节过滤的）用例。"""
    selected = _select(chapters)
    gpu = has_gpu()
    results: list[Result] = []

    for case in selected:
        if case.gpu and not gpu:
            results.append(Result(case, SKIP, "需要 GPU（当前不可见）"))
            continue
        try:
            case.fn()
        except Exception as exc:  # noqa: BLE001 - 测试失败就是失败，不区分异常类型
            results.append(Result(case, FAIL, f"{type(exc).__name__}: {exc}"))
        else:
            results.append(Result(case, PASS))

    return results


def _select(chapters: Iterable[str] | None) -> list[Case]:
    if not chapters:
        return list(_CASES)
    wanted = tuple(chapters)
    out = []
    for case in _CASES:
        if case.group == "meta" or any(w in case.group for w in wanted):
            out.append(case)
    return out


# -------------------------------------------------------------------- 报告


def report(results: list[Result], verbose: bool = False) -> int:
    """打印结果，返回失败数。

    默认只详列**失败**的用例。跳过的不逐条打印 —— 没有显卡时会有几十条跳过，
    把失败信息淹没掉才是真的危险。加 -v 才逐个列出。
    """
    if not results:
        print("没有匹配的测试用例")
        return 0

    passed = [r for r in results if r.status == PASS]
    skipped = [r for r in results if r.status == SKIP]
    failures = [r for r in results if r.status == FAIL]

    if verbose:
        for r in passed:
            print(f"  PASS  {r.case.group}/{r.case.name}")
        for r in skipped:
            print(f"  SKIP  {r.case.group}/{r.case.name}  {r.detail}")

    # 失败放最后、带缩进的详情，方便直接看原因
    for r in failures:
        print(f"  FAIL  {r.case.group}/{r.case.name}")
        print(f"        {r.detail}")

    print("-" * 68)
    print(f"合计 {len(results)}：通过 {len(passed)}，失败 {len(failures)}，跳过 {len(skipped)}")

    if skipped and not verbose:
        counts: dict[str, int] = {}
        for r in skipped:
            counts[r.case.group] = counts.get(r.case.group, 0) + 1
        detail = "、".join(f"{g} {n} 个" for g, n in sorted(counts.items()))
        print(f"跳过明细：{detail}")
    if skipped and not has_gpu():
        print("（当前看不到 GPU，跳过的都是需要显卡的用例；在有显卡的机器上重跑即可覆盖）")

    return len(failures)
