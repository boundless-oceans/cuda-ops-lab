"""第 5 章特有的边界测试（需要 GPU）。

通用正确性由 `test_generic.py` 覆盖（每个变体 × 全部 EDGE_SHAPES，判据是**精确相等**）。
这里补的是转置特有的几件事：

## 1. 用"编号矩阵"把每个元素的位置钉死

输入用 `arange(R*C).reshape(R,C)`（元素值就是它的输入下标）。转置之后，
`out[c][r]` 必须恰好等于 `r*C + c`。**任何"搬错位置"都会立刻暴露**，
而且错误信息里能看出是哪个下标错了 —— 比随机数好查得多。

## 2. 瓦片接缝

M 或 N 不是 32 的倍数时，最后一行/列的瓦片是残的。残瓦片的两个方向边界条件
是**交叉**的（`out_row+j < cols` 对 `in_col+k < cols`），写错一边就会读到没写过的
共享内存格子。`(1,N)`、`(N,1)`、`(31,33)`、`(33,31)` 这几个形状专门打它。

## 3. `x.t()` 当输入（非连续）

转置算子最容易踩的一个坑：`x.t()` 本身是**视图**。绑定层不先 `.contiguous()`
就取 `data_ptr()`，会把"逻辑上的转置"当成"内存里的转置"，于是**又转回去了** ——
结果是一个看似合理但完全错的矩阵。
"""

from __future__ import annotations

import torch

from harness import assert_true, test
from ops_lab import registry as reg

GROUP = "05_transpose"


def _variants() -> list[reg.VariantRef]:
    return [r for r in reg.iter_variants() if r.chapter == GROUP]


@test("编号矩阵：每个元素的位置都精确可查", group=GROUP, gpu=True)
def test_numbered_matrix() -> None:
    shapes = [(1, 1), (1, 17), (17, 1), (31, 33), (32, 32), (33, 31), (33, 33), (64, 64),
              (100, 130)]
    for rows, cols in shapes:
        x = torch.arange(rows * cols, dtype=torch.float32, device="cuda").reshape(rows, cols)
        # out[c][r] 必须等于 r*cols + c
        expected = (torch.arange(rows, device="cuda").reshape(1, rows) * cols
                    + torch.arange(cols, device="cuda").reshape(cols, 1)).double()
        for ref in _variants():
            got = reg.call(ref, x)
            assert_true(tuple(got.shape) == (cols, rows),
                        f"{ref}: 输出形状应为 {(cols, rows)}，实际 {tuple(got.shape)}")
            assert_true(bool(torch.equal(got.double(), expected)),
                        f"{ref} ({rows}×{cols}): 编号矩阵转置后元素位置不对")


@test("空矩阵：输出形状是转置后的形状", group=GROUP, gpu=True)
def test_empty_matrix() -> None:
    for rows, cols in [(0, 0), (0, 7), (7, 0)]:
        x = torch.empty(rows, cols, dtype=torch.float32, device="cuda")
        for ref in _variants():
            out = reg.call(ref, x)
            assert_true(tuple(out.shape) == (cols, rows),
                        f"{ref}: ({rows},{cols}) 的输出形状应为 {(cols, rows)}，"
                        f"实际 {tuple(out.shape)}")
            assert_true(out.numel() == 0, f"{ref}: 空矩阵输出元素数应为 0")


@test("输入是 x.t()（非连续）时结果仍然正确", group=GROUP, gpu=True)
def test_non_contiguous_view() -> None:
    """`x.t()` 是视图。绑定层必须 `.contiguous()` 之后再取指针 ——
    否则会把"逻辑上的转置"当成"内存里的转置"，**又转回去**。"""
    base = torch.randn(64, 48, dtype=torch.float32, device="cuda")
    x = base.t()                       # (48, 64) 的视图，非连续
    assert_true(not x.is_contiguous(), "测试前提不成立：转置视图应当是连续的")

    expected = x.double().transpose(0, 1).contiguous()
    for ref in _variants():
        got = reg.call(ref, x)
        assert_true(tuple(got.shape) == (64, 48),
                    f"{ref}: 输出形状应为 (64, 48)，实际 {tuple(got.shape)}")
        assert_true(bool(torch.equal(got.double(), expected)),
                    f"{ref}: 非连续输入的结果不对（是不是忘了 .contiguous()？）")


@test("非 2 维输入抛清晰错误", group=GROUP, gpu=True)
def test_dim_check() -> None:
    for bad in (torch.randn(16, device="cuda"), torch.randn(4, 4, 4, device="cuda")):
        try:
            reg.call(_variants()[0], bad)
        except RuntimeError as exc:
            assert_true("2 维" in str(exc), f"错误信息里应提到「2 维」，实际是：{exc}")
        else:
            raise AssertionError(f"{bad.dim()} 维输入应当抛 RuntimeError，但没有")


@test("dtype 不是 float32 时抛清晰错误", group=GROUP, gpu=True)
def test_dtype_check() -> None:
    x = torch.randn(8, 8, dtype=torch.float64, device="cuda")
    try:
        reg.call(_variants()[0], x)
    except RuntimeError as exc:
        assert_true("float32" in str(exc), f"错误信息里应提到 float32，实际是：{exc}")
    else:
        raise AssertionError("dtype 为 float64 时应当抛 RuntimeError，但没有")
