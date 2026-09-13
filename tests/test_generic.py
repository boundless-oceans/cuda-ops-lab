"""数据驱动的数值正确性测试（需要 GPU）。

对每个 (章节, 算子, 变体) 注册一个用例，内部遍历该算子的全部测试形状。

## 比对方法（第 2 步是关键）

    1. 用 float32 造输入 —— 这就是 kernel 要吃的值
    2. 把**同样的值**升成 float64，交给参考实现算真值
    3. kernel 跑一遍，按 |got - ref| <= atol + rtol * |ref| 比对

第 2 步容易做错：如果参考实现另生成一组 float64 随机数，比对结果里就会掺进
"输入从 float64 降成 float32"的量化误差（约 1e-7 相对），于是容差设成多少
都说不清。用同一组值就没这个问题，剩下的误差全部来自 kernel。

## 为什么一个变体只注册一个用例

15 个变体 × 9 个形状 = 135 个用例，报告会被淹没。所以按变体聚合，
失败信息里带上具体形状 —— 定位能力一样，输出清爽得多。
"""

from __future__ import annotations

import torch

from harness import assert_close, assert_true, test
from ops_lab import registry as reg

# 固定随机种子：同一个形状每次跑都用同一组输入，失败可复现
_SEED = 20260913


def _check_shape(spec: dict, ref: reg.VariantRef, shape: tuple) -> None:
    shape = tuple(shape)
    torch.manual_seed(_SEED)

    n_in = reg.arity(spec)
    gen = spec.get("gen")
    if gen is not None:
        inputs32 = gen(shape)
    else:
        inputs32 = tuple(torch.randn(shape, dtype=torch.float32) for _ in range(n_in))

    # 参考实现吃的是"同样的值升上来的 float64"
    inputs64 = tuple(t.double() if torch.is_tensor(t) else t for t in inputs32)
    expected = spec["reference"](*inputs64)

    device = torch.device("cuda")
    kernel_inputs = tuple(t.to(device) if torch.is_tensor(t) else t for t in inputs32)

    got = reg.call(ref, *kernel_inputs)

    assert_true(torch.is_tensor(got), f"{ref} 没有返回张量，实际返回 {type(got).__name__}")
    assert_true(bool(got.is_cuda), f"{ref} 的返回值不在 CUDA 上")

    assert_close(
        got,
        expected,
        rtol=spec.get("rtol", 1e-5),
        atol=spec.get("atol", 1e-6),
        what=f"[shape={shape}] ",
    )


def _make_case(spec: dict, ref: reg.VariantRef):
    def run() -> None:
        for shape in spec["shapes"]:
            _check_shape(spec, ref, shape)

    return run


# 按元数据自动注册全部用例 —— 新增算子/变体不需要改这个文件
for _chapter, _op, _spec in reg.iter_ops():
    for _label in _spec["variants"]:
        _ref = reg.find_variant(_chapter, _op, _label)
        assert _ref is not None, f"元数据不一致：找不到 {_chapter}/{_op}/{_label}"
        test(name=f"{_op}/{_label}", group=_chapter, gpu=True)(_make_case(_spec, _ref))
