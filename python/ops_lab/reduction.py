"""第 3 章：reduction（sum / max 的五级优化阶梯）。

两个算子的 kernel 结构完全一样，只有 combine 与单位元不同（见
`kernels/03_reduction/reduction_ops.cuh` 的 SumOp / MaxOp），所以这里也用循环
生成变体表，而不是手抄两遍。

## 本章的元数据和第 2 章有两处结构性差异

1. **输出形状不再等于输入形状**：永远是 `(1,)`。所以描述符里多了一个
   `out_shape` 字段 —— `tests/test_metadata.py` 用它来校验参考实现的输出形状，
   不然那条"输出形状应与输入一致"的检查在归约上必然误报。

2. **容差是逐算子给的，不是全局默认值**：`max` 是纯比较、没有算术，可以要求
   **精确相等**（rtol = atol = 0）；`sum` 是浮点累加，误差随累加深度增长，
   得给一个说得清来源的容差（见下面 `rtol` 的注释）。
"""

from __future__ import annotations

import torch

NAME = "03_reduction"

# 阶梯顺序。基准脚本按这个顺序排表；字典序与它一致（v0 < v1 < … < v4），
# 但显式写出来才不会被后来的命名打乱。
VARIANT_ORDER = (
    "v0_uncoalesced",
    "v1_naive",
    "v2_smem_tree",
    "v3_warp_shuffle",
    "v4_single_block",
)

VARIANT_NOTES = {
    "v0_uncoalesced": "反面教材：相邻线程地址相隔 gridDim.x，一个 warp 跨 32 条 cache line",
    "v1_naive": "grid-stride 合并访存 + 块内 smem 顺序寻址树 + 每 block 一次原子合并",
    "v2_smem_tree": "smem 按 warp 加 padding（tid → tid + tid/32）；顺序寻址本无冲突，预期无差别",
    "v3_warp_shuffle": "块内改用 __shfl_down_sync，共享内存访问 8×256 次 → 8 次，同步 8 次 → 1 次",
    "v4_single_block": "故意回退：块内沿用 v3 的写法，但 grid = 1，只用了 1/24 的 SM",
}

# 形状必须覆盖这些情况：
#   0            空输入。**归约特有的**：输出仍要写出单位元（sum→0，max→-inf），
#                而主 kernel 根本不启动 —— 这条路径只能靠 n=0 测出来
#   1            极小；只有 1 个线程有元素，其余 255 个必须贡献单位元
#   31/32/33     一个 warp 的三个位置。33 那一条专门测"跨 warp 的第一个元素"
#   255/256/257  一个 block 的三个位置
#   1024         多个 block，且是 2 的幂
#   1000003      大、奇数、非 2 的幂 —— 三个不利条件叠在一起
#
# 注意这里**故意没有**把形状造得很大：N = 2^24 时漏算一个元素只让 sum 差 6e-8
# 相对误差，比容差还小，等于测不出来。"分区恰好覆盖一次"这件事必须靠小 N
# 且卡在边界上的形状来查，大形状负责查的是别的东西（不崩、不超时、数值不漂）。
EDGE_SHAPES = [(0,), (1,), (31,), (32,), (33,), (255,), (256,), (257,), (1024,), (1000003,)]

# 基准专用形状 —— **必须和测试形状分开**（第 2 章踩过的坑：测试形状的工作集
# 装得进 L2，测出来的是 L2 带宽而不是显存带宽）。
#
# 2^24 = 16.7M 个元素 → 读约 67.1 MB，远超 L2；按 256 GB/s 估算理论耗时
# 约 262 µs，长到能让计时误差可以忽略。
BENCH_SHAPES = [(1 << 24,)]

# 测试里用来推算 grid-stride 步长的 block 大小。
# **必须与 `kernels/03_reduction/reduction_block.cuh` 的 kBlockSize 一致**，
# `tests/test_03_reduction_edge.py` 用它构造"恰好卡在循环边界上"的形状。
BLOCK_SIZE = 256


def _variants(op: str) -> dict[str, str]:
    """一变体一符号：reduction_<算子>_<变体>。"""
    return {label: f"reduction_{op}_{label}" for label in VARIANT_ORDER}


def _sum_reference(x):
    """参考实现：在 float64 上求和，输出形状固定为 (1,)。"""
    return x.sum().reshape(1)


def _max_reference(x):
    """参考实现：在 float64 上取最大，输出形状固定为 (1,)。

    torch 对空张量求最大值会直接抛异常，而"空数组的最大值"在本章的 kernel 里
    有明确定义 —— 单位元 -inf（init kernel 写的就是它）。参考实现必须给出同一个
    值，否则这个边界根本无从比对。
    """
    if x.numel() == 0:
        return x.new_full((1,), float("-inf"))
    return x.max().reshape(1)


OPS = {
    "sum": {
        "variants": _variants("sum"),
        "reference": _sum_reference,
        "shapes": EDGE_SHAPES,
        "bench_shapes": BENCH_SHAPES,
        "out_shape": lambda shape: (1,),
        # 全正数输入。用 rand 而不是 randn 是刻意的：randn 在 N 很大时求和会
        # 出现正负相消，结果的相对误差失去意义，容差就变成了玄学。
        "gen": lambda shape: (torch.rand(shape, dtype=torch.float32),),
        # 读 4N + 写 4（输出只有一个数）。写的这 4 个字节不能省：
        # 它虽然对带宽没有影响，但决定了"算术强度"这一列到底是多少。
        "bytes": lambda shape: shape[0] * 4 + 4,
        "flops": lambda shape: max(shape[0] - 1, 0),
        # 归约是浮点累加，误差随累加深度增长。上界按最坏情况估：
        # 每线程串行累加 ~K 项（K = N / 线程数）→ 误差 ~K·eps，块内树再加 8 级，
        # 144 个部分和之间还有一层。N ≈ 1e6、K ≈ 27 时最坏约 1e-4 相对；
        # 实际（随机舍入）一般在 1e-6 量级。
        #
        # 取 1e-4 是"能挡住真 bug、又不误报"的量级：漏算/多算**一整块**
        # 会让结果差千分之几以上，远在容差之外；而卡边界的小形状（N ≤ 257）
        # 哪怕只漏一个元素也有 0.4% 的偏差，同样跑不掉。
        "rtol": 1e-4,
        "atol": 1e-6,
        "torch": _sum_reference,
    },
    "max": {
        "variants": _variants("max"),
        "reference": _max_reference,
        "shapes": EDGE_SHAPES,
        "bench_shapes": BENCH_SHAPES,
        "out_shape": lambda shape: (1,),
        # 带负数的输入。这是为了测**单位元**：如果 max 的单位元错写成 0.0f，
        # 全正数输入下一点问题都没有，只有输入里有负数时才会暴露。
        "gen": lambda shape: (torch.randn(shape, dtype=torch.float32),),
        "bytes": lambda shape: shape[0] * 4 + 4,
        "flops": lambda shape: max(shape[0] - 1, 0),
        # max 只有比较、没有算术，结果是**精确**的，所以要求精确相等。
        # 这比给一个 1e-5 的容差强得多：任何"少算一个元素"都会被抓到，
        # 而那些元素恰好是最大值时更是必抓。
        "rtol": 0.0,
        "atol": 0.0,
        "torch": _max_reference,
    },
}
