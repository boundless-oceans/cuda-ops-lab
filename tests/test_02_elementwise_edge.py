"""第 2 章特有的边界测试（需要 GPU）。

通用正确性由 `test_generic.py` 覆盖。这里补的是**元数据描述不到**的那些行为：
空张量、非连续输入、多维形状、以及参数校验是否真的会拦下错误输入。

## 一个已知的测试盲区

`grid_for()` 的溢出分支（所需 block 数超过 gridDim.x 上限 2^31-1）**无法在真机上
测试** —— 触发它需要 n > 2^31-1 × 256 ≈ 5.5e11 个元素，也就是 2 TB 显存。
所以那个分支只能靠代码审查保证。写在这里是为了让后来的人知道：
"没测到"不等于"没问题"。
"""

from __future__ import annotations

import torch

from harness import assert_close, assert_true, test
from ops_lab import _extension

GROUP = "02_elementwise"

# 每个算子的 (扩展函数名, 输入个数)
_CASES = {
    "add": ("elementwise_add_v1_naive", 2),
    "relu": ("elementwise_relu_v1_naive", 1),
    "sigmoid": ("elementwise_sigmoid_v1_naive", 1),
}


def _fn(op: str):
    name, _ = _CASES[op]
    return getattr(_extension.get(), name)


@test("空张量返回空结果且不报错", group=GROUP, gpu=True)
def test_empty_tensor() -> None:
    """n == 0 时绑定层应当直接返回空张量，**不去启动一个 grid == 0 的 kernel**。"""
    for op, (_, arity) in _CASES.items():
        args = tuple(torch.empty(0, device="cuda") for _ in range(arity))
        out = _fn(op)(*args)
        assert_true(out.numel() == 0, f"{op}: 空输入的输出元素数应为 0，实际 {out.numel()}")
        assert_true(out.is_cuda, f"{op}: 空输入的输出仍应在 CUDA 上")
        assert_true(out.dtype == torch.float32, f"{op}: 空输入的输出 dtype 应为 float32")


@test("非连续输入会先转连续，结果正确", group=GROUP, gpu=True)
def test_non_contiguous() -> None:
    """切片得到的张量是非连续的。绑定层做 `.contiguous()` 后再取 data_ptr()，
    否则拿到的是错的内存、**结果会静默算错**。"""
    base = torch.randn(2048, dtype=torch.float32, device="cuda")
    x = base[::2]           # 非连续
    y = base[1::2]          # 非连续，长度相同
    assert_true(not x.is_contiguous(), "测试前提不成立：切片结果应当是连续的")

    got = _fn("add")(x, y)
    expected = x.double() + y.double()
    assert_close(got, expected, 1e-5, 1e-6, what="非连续输入 ")


@test("多维输入按 numel 展平处理", group=GROUP, gpu=True)
def test_multidim() -> None:
    """元数据里的形状都是 1 维，但 elementwise 应当对任意形状成立
    （kernel 只认 numel 和连续内存）。"""
    shape = (17, 33, 7)     # 刻意都不整齐
    x = torch.randn(shape, dtype=torch.float32, device="cuda")
    y = torch.randn(shape, dtype=torch.float32, device="cuda")

    got = _fn("add")(x, y)
    assert_true(tuple(got.shape) == shape, f"输出形状应为 {shape}，实际 {tuple(got.shape)}")
    assert_close(got, x.double() + y.double(), 1e-5, 1e-6, what="三维输入 ")


@test("shape 不一致时抛清晰错误", group=GROUP, gpu=True)
def test_shape_mismatch_raises() -> None:
    x = torch.randn(128, dtype=torch.float32, device="cuda")
    y = torch.randn(64, dtype=torch.float32, device="cuda")
    try:
        _fn("add")(x, y)
    except RuntimeError as exc:
        assert_true("shape" in str(exc), f"错误信息里应提到 shape，实际是：{exc}")
    else:
        raise AssertionError("shape 不一致时应当抛 RuntimeError，但没有")


@test("dtype 不是 float32 时抛清晰错误", group=GROUP, gpu=True)
def test_dtype_check() -> None:
    x = torch.randn(128, dtype=torch.float64, device="cuda")
    try:
        _fn("relu")(x)
    except RuntimeError as exc:
        assert_true("float32" in str(exc), f"错误信息里应提到 float32，实际是：{exc}")
    else:
        raise AssertionError("dtype 为 float64 时应当抛 RuntimeError，但没有")
