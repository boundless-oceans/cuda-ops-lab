"""第 4 章特有的边界测试（需要 GPU）。

通用正确性（每个变体 × 全部 EDGE_SHAPES）由 `test_generic.py` 自动覆盖。
这里补的是 scan **特有**的几件事，而第 1 条是整章最锋利的检查：

## 1. 全 1 输入 → 输出必须**精确**等于 1,2,3,…,n

前缀和最容易出的错是"少算/多算一段"（chunk 边界、尾部补齐、block 前缀加错）。
用全 1 输入时答案恰好是 `i+1`，而 `i+1` 在 float32 里可以**精确表示**（n ≪ 2²⁴），
所以这里可以要求 `rtol = atol = 0`：

  * 漏掉一个元素 → 后面所有输出差 1
  * 多加一段 → 后面所有输出差一个整数
  * 前缀和跨 chunk 传递错了 → 从某个 chunk 开始整体偏移

**这些误差在随机输入下会被 1e-3 的容差吃掉一部分；在全 1 输入下无处可藏。**
这就是为什么容差可以放宽、而边界测试不能省。

## 2. block 前缀的"搭便车"路径：n 恰好跨过可驻留容量

单趟变体（v3）的 grid 被截到"可同时驻留的 block 数"，所以 n 一旦超过
`容量 × 256`，每个 block 就要处理**多个 chunk**，look-back 必须跨轮次工作。
构造 n = stride、stride+1（stride = 容量 × 256）正好卡在那条边界上。

## 3. 展平语义

kernel 只认 numel 和连续内存，所以多维输入按**展平**处理（与第 2、3 章一致）。
逐行扫描是另一类算子（softmax / norm），第 6、7 章才讲 —— 这里显式钉住当前语义。
"""

from __future__ import annotations

import torch

from harness import assert_close, assert_true, test
from ops_lab import device_query
from ops_lab import registry as reg

GROUP = "04_scan"

# 与 kernels/04_scan/scan_config.cuh 的 kChunkSize / kOptimizedItemsPerThread 一致
CHUNK = 256
OPTIMIZED_ITEMS = 8


def _variants() -> list[reg.VariantRef]:
    return [r for r in reg.iter_variants() if r.chapter == GROUP]


def _capacity_blocks() -> int:
    """可同时驻留的 block 数（= SM 数 × 每 SM 块数）。"""
    info = device_query.device_info() or {}
    sm = int(info.get("sm_count") or 0)
    threads_per_sm = int(info.get("max_threads_per_sm") or 0)
    if sm <= 0 or threads_per_sm <= 0:
        return 1  # 探测不到就退化成"一个 block"
    return sm * max(threads_per_sm // CHUNK, 1)


def _strides() -> list[int]:
    """两条容量边界：K=1 的变体（chunk 256）与 K=8 的变体（chunk 2048）。

    超过 `容量 × chunk 大小`，每个 block 就要处理多个 chunk，
    look-back 必须跨轮次工作 —— 这是单趟实现最容易错的地方。
    """
    cap = _capacity_blocks()
    return [cap * CHUNK, cap * CHUNK * OPTIMIZED_ITEMS]


@test("全 1 输入 → 前缀和精确等于 1,2,3,…（最锋利的检查）", group=GROUP, gpu=True)
def test_ones_exact() -> None:
    shapes = [1, 2, 31, 32, 33, 255, 256, 257, 511, 512, 513, 1024, 4096, 1000003]

    for n in shapes:
        x = torch.ones(n, dtype=torch.float32, device="cuda")
        expected = torch.arange(1, n + 1, dtype=torch.float64)
        for ref in _variants():
            got = reg.call(ref, x)
            assert_close(got, expected, 0.0, 0.0, what=f"{ref} n={n} 全 1 输入")


@test("跨 chunk 的前缀传递：卡在容量边界上", group=GROUP, gpu=True)
def test_capacity_boundary() -> None:
    """n = stride−1 / stride / stride+1 / 2·stride+1，**两条 stride 都测**。

    stride = 可驻留 block 数 × chunk 大小，而 chunk 大小随变体不同
    （K=1 是 256，K=8 是 2048）。这四个形状分别对应：
        不足一轮（有 block 一个 chunk 都没有）
        恰好一轮
        有 block 要处理第二个 chunk（look-back 必须跨轮次）
        两轮多一个
    """
    for stride in _strides():
        for n in (stride - 1, stride, stride + 1, 2 * stride + 1):
            x = torch.ones(n, dtype=torch.float32, device="cuda")
            expected = torch.arange(1, n + 1, dtype=torch.float64)
            for ref in _variants():
                got = reg.call(ref, x)
                assert_close(got, expected, 0.0, 0.0, what=f"{ref} n={n}（stride={stride}）")


@test("空输入返回空输出", group=GROUP, gpu=True)
def test_empty() -> None:
    x = torch.empty(0, dtype=torch.float32, device="cuda")
    for ref in _variants():
        out = reg.call(ref, x)
        assert_true(out.numel() == 0, f"{ref}: 空输入输出元素数应为 0，实际 {out.numel()}")
        assert_true(out.is_cuda, f"{ref}: 空输入输出仍应在 CUDA 上")


@test("非连续输入会先转连续，结果正确", group=GROUP, gpu=True)
def test_non_contiguous() -> None:
    """切片得到的张量是非连续的。绑定层做 `.contiguous()` 后再取 data_ptr()，
    否则前缀和会从错位的内存上算起 —— 而且**不会报错**，只会整体偏移。"""
    base = torch.randn(4096, dtype=torch.float32, device="cuda")
    x = base[::2]
    assert_true(not x.is_contiguous(), "测试前提不成立：切片结果应当是连续的")

    expected = x.double().cumsum(0)
    for ref in _variants():
        got = reg.call(ref, x)
        assert_close(got, expected, 1e-3, 1e-4, what=f"{ref} 非连续输入 ")


@test("多维输入按 numel 展平（显式钉住这个语义）", group=GROUP, gpu=True)
def test_multidim_flattened() -> None:
    """kernel 只认 numel 和连续内存 → 多维输入是**展平后**做一次前缀和，
    不是逐行扫。逐行是 softmax/norm 那一路（第 6、7 章）。"""
    shape = (17, 33, 7)  # 3927 个元素，刻意都不整齐
    x = torch.randn(shape, dtype=torch.float32, device="cuda")
    expected = x.double().reshape(-1).cumsum(0).reshape(shape)

    for ref in _variants():
        got = reg.call(ref, x)
        assert_true(tuple(got.shape) == shape,
                    f"{ref}: 输出形状应与输入一致 {shape}，实际 {tuple(got.shape)}")
        assert_close(got, expected, 1e-3, 1e-4, what=f"{ref} 三维展平 ")


@test("dtype 不是 float32 时抛清晰错误", group=GROUP, gpu=True)
def test_dtype_check() -> None:
    x = torch.randn(128, dtype=torch.float64, device="cuda")
    try:
        reg.call(_variants()[0], x)
    except RuntimeError as exc:
        assert_true("float32" in str(exc), f"错误信息里应提到 float32，实际是：{exc}")
    else:
        raise AssertionError("dtype 为 float64 时应当抛 RuntimeError，但没有")
