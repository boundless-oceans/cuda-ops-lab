"""第 3 章特有的边界测试（需要 GPU）。

通用正确性（每个变体 × 全部 EDGE_SHAPES）由 `test_generic.py` 自动覆盖。
这里补的是元数据描述不到的那些行为：

  * 空输入必须写出**单位元**（sum → 0，max → -inf），而不是未初始化内存
  * 非连续输入、多维输入
  * **grid-stride 的分区边界**：元素是否恰好被覆盖一次
  * dtype 校验

## 两个盲区，写在这里让后来的人知道"没测到"不等于"没问题"

1. **大 N 会掩盖单元素错误。** N = 2^24 时漏算一个元素只让 sum 差 6e-8 相对误差，
   比容差还小 —— 这类 bug 在大形状上永远测不出来。"分区恰好覆盖一次"必须靠
   **小 N 且卡在循环边界上**的形状来查，这正是 `test_grid_stride_partition` 干的事：
   它用**全 1 输入**，于是 float32 的求和是**精确**的（部分和都是 ≤ 7e4 的整数，
   在 float32 里可以精确表示），可以要求 `rtol = atol = 0`。任何多算/漏算一个
   元素都会让结果差一个整数，无处可藏。

2. **v4 的"慢"这里不做断言。** 单 block 变体是故意留的回退级，但把"它比 v1 慢
   10 倍"这种性能断言写进正确性测试，会让测试依赖具体机器。性能由
   `bench/run_bench.py` 的阶梯表负责，这里只保证它**算得对**。

## 一个已知的运行期盲区

`init_reduction_output` 用的那个 1 线程 kernel，其**启动失败**在这里测不到：
grid=1、block=1 的启动参数永远合法。它出问题只可能是 CUDA 上下文已经崩了，
那种情况下后面任何一条测试都会失败。
"""

from __future__ import annotations

import torch

from harness import assert_close, assert_true, test
from ops_lab import device_query
from ops_lab import registry as reg

GROUP = "03_reduction"

# 与 `python/ops_lab/reduction.py` 的 BLOCK_SIZE、
# `kernels/03_reduction/reduction_block.cuh` 的 kBlockSize 保持一致。
# 只用来推算 grid-stride 的步长；万一哪天三处不同步，最坏的后果是这个测试
# 卡不到精确的边界上（形状仍然合法、检查仍然有效），不会误报。
BLOCK = 256


def _variants(op: str) -> list[reg.VariantRef]:
    return [r for r in reg.iter_variants() if r.chapter == GROUP and r.op == op]


def _grid_stride() -> int:
    """v0~v3 的 grid-stride 步长 = grid × block = min(覆盖 n 所需, 填满机器的) × block。

    取设备真实的 SM 数与每 SM 线程数来算，而不是写死 36864 —— 换个卡这个测试
    仍然卡在正确的边界上。
    """
    info = device_query.device_info() or {}
    sm = int(info.get("sm_count") or 0)
    threads_per_sm = int(info.get("max_threads_per_sm") or 0)
    if sm <= 0 or threads_per_sm <= 0:
        # 探测不到就退化成"一个 block"：至少还能覆盖块内分区那一段边界
        return BLOCK
    return sm * max(threads_per_sm // BLOCK, 1) * BLOCK


@test("空输入返回单位元而不是未初始化内存", group=GROUP, gpu=True)
def test_empty_returns_identity() -> None:
    """n == 0 时主 kernel 不启动。此时输出必须已经是单位元。

    为什么这条特别值得单独测：`torch::empty({1})` 里是**未初始化内存**，
    如果 launcher 的空输入分支把 init kernel 也一起跳过，结果会是一个
    随机的数 —— 而"随机数"在多数形状下都不会引起注意，只有恰好测 n=0 才看得见。
    """
    expectations = {"sum": 0.0, "max": float("-inf")}
    for op, want in expectations.items():
        for ref in _variants(op):
            x = torch.empty(0, dtype=torch.float32, device="cuda")
            out = reg.call(ref, x)

            assert_true(
                tuple(out.shape) == (1,), f"{ref}: 空输入输出形状应为 (1,)，实际 {tuple(out.shape)}"
            )
            assert_true(bool(out.is_cuda), f"{ref}: 空输入输出仍应在 CUDA 上")
            assert_true(out.dtype == torch.float32, f"{ref}: 输出 dtype 应为 float32")
            got = float(out.item())
            assert_true(
                got == want,
                f"{ref}: 空输入应返回单位元 {want!r}，实际 {got!r}（未初始化的内存？）",
            )


@test("非连续输入会先转连续，结果正确", group=GROUP, gpu=True)
def test_non_contiguous() -> None:
    """切片得到的张量是非连续的。绑定层做 `.contiguous()` 后再取 data_ptr()，
    否则归约会读一串错位的内存 —— 而且**不会报错**，只会给出一个偏大或偏小的数。"""
    base = torch.randn(4096, dtype=torch.float32, device="cuda")
    x = base[::2]
    assert_true(not x.is_contiguous(), "测试前提不成立：切片结果应当是连续的")

    for op in ("sum", "max"):
        for ref in _variants(op):
            got = reg.call(ref, x)
            expected = x.double().sum().reshape(1) if op == "sum" else x.double().max().reshape(1)
            assert_close(got, expected, 1e-4, 1e-6, what=f"{ref} 非连续输入 ")


@test("多维输入按 numel 展平，输出仍是 (1,)", group=GROUP, gpu=True)
def test_multidim() -> None:
    """元数据里的形状都是 1 维，但归约应当对任意形状成立
    （kernel 只认 numel 和连续内存）。"""
    shape = (17, 33, 7)  # 刻意都不整齐：3927 个元素
    x = torch.randn(shape, dtype=torch.float32, device="cuda")

    for op in ("sum", "max"):
        for ref in _variants(op):
            got = reg.call(ref, x)
            assert_true(
                tuple(got.shape) == (1,),
                f"{ref}: 三维输入应当展平后归约成 (1,)，实际 {tuple(got.shape)}",
            )
            expected = x.double().sum().reshape(1) if op == "sum" else x.double().max().reshape(1)
            assert_close(got, expected, 1e-4, 1e-6, what=f"{ref} 三维输入 ")


@test("grid-stride 分区恰好覆盖每个元素一次", group=GROUP, gpu=True)
def test_grid_stride_partition() -> None:
    """把形状卡在循环步长 stride 的边界上：stride-1 / stride / stride+1 / 2·stride / 2·stride+1。

    这五个形状分别对应：
        stride-1      所有线程都跑不满一轮（有线程一个元素都没有）
        stride        每个线程恰好一个元素
        stride+1      有线程拿到第二个元素
        2·stride      每个线程恰好两个元素
        2·stride+1    最后一个不完整的第三轮

    输入用全 1（查 sum）与 arange（查 max），两者都让结果**精确可算**，
    所以这里可以要求精确相等 —— 漏算或多算任何一个元素都会立刻暴露。
    这是本章最有价值的一条测试：错误的索引公式（比如 v0 那行写错一点）
    在大形状上可能只差 1e-7，在这里必定差一个整数。
    """
    stride = _grid_stride()
    shapes = [stride - 1, stride, stride + 1, 2 * stride, 2 * stride + 1]

    for n in shapes:
        ones = torch.ones(n, dtype=torch.float32, device="cuda")
        ramp = torch.arange(n, dtype=torch.float32, device="cuda")

        for ref in _variants("sum"):
            got = reg.call(ref, ones)
            assert_close(
                got,
                torch.tensor([float(n)], dtype=torch.float64),
                0.0,
                0.0,
                what=f"{ref} n={n}（全 1 求和应为精确的 {n}）",
            )

        for ref in _variants("max"):
            got = reg.call(ref, ramp)
            assert_close(
                got,
                torch.tensor([float(n - 1)], dtype=torch.float64),
                0.0,
                0.0,
                what=f"{ref} n={n}（arange 最大值应为精确的 {n - 1}）",
            )


@test("dtype 不是 float32 时抛清晰错误", group=GROUP, gpu=True)
def test_dtype_check() -> None:
    x = torch.randn(128, dtype=torch.float64, device="cuda")
    ref = _variants("sum")[0]
    try:
        reg.call(ref, x)
    except RuntimeError as exc:
        assert_true("float32" in str(exc), f"错误信息里应提到 float32，实际是：{exc}")
    else:
        raise AssertionError("dtype 为 float64 时应当抛 RuntimeError，但没有")
