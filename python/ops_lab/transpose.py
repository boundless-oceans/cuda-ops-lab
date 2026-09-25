"""第 5 章：transpose（二维转置的六级阶梯）。

## 这一章的元数据有三个"第一次"

1. **形状是二维的**（`(rows, cols)`），而前面几章全是一维。于是：
   * `out_shape` 第一次真的用上（输出形状 = `(cols, rows)`）——第 3 章加的字段；
   * `test_metadata.py` 里"形状必须覆盖 n=0 与 n=1"那条检查要改成按**元素个数**判
     （二维里没有 `(0,)` / `(1,)` 这种元组）。
2. **容差是精确相等**（`rtol = atol = 0`）：转置是**纯置换**，一个算术运算都没有，
   所以结果必须逐位相同。这比给 1e-5 的容差强得多 —— 任何"搬错一个元素"都会被抓到。
3. **`flops = 0`**：本章不做任何算术，算术强度这一列恒为 0（它就是纯数据搬运的极端）。

`bytes` = 读 4MN + 写 4MN = **8MN**，与第 2、4 章同一个访存形态 ——
所以本章的成绩可以直接放进"访存形态 vs %峰值"那张横向对比表。
"""

from __future__ import annotations

import torch

NAME = "05_transpose"

VARIANT_ORDER = (
    "v0_naive",
    "v1_swapped",
    "v2_smem_tiled",
    "v3_padded",
    "v4_vectorized",
    "v5_wide_tile",
)

VARIANT_NOTES = {
    "v0_naive": "反面教材：读合并、写跨步（相邻线程隔 rows 个元素）",
    "v1_swapped": "反面教材：读跨步、写合并；与 v0 只差交换 (r,c) 的取法",
    "v2_smem_tiled": "32×32 smem 瓦片：两边都合并，但列访问 32 路 bank conflict",
    "v3_padded": "与 v2 只差共享内存行宽 32→33（消掉 32 路冲突）",
    "v4_vectorized": "与 v3 只差访存宽度：读写都用 float4（线程映射随之改变）",
    "v5_wide_tile": "与 v4 只差瓦片宽度 32→64：每次连续访问 128 B → 256 B",
}

# 形状必须覆盖这些情况：
#   (0,*) / (*,0)  空矩阵（输出是转置后的形状，0 个元素）
#   (1,1)          单元素
#   (1,N) / (N,1)  退化的长条 —— 瓦片在这里完全对不上，最容易漏
#   31 / 32 / 33   瓦片边界的三个位置（两个方向都要）
#   64             恰好两个瓦片
#   255×257        大、都不整齐、非 2 的幂
#   1000×1001      更不整齐
EDGE_SHAPES = [(0, 0), (0, 5), (5, 0), (1, 1), (1, 1024), (1024, 1), (31, 32), (32, 32),
               (32, 33), (33, 32), (33, 33), (64, 64), (255, 257), (1000, 1001)]

# 基准专用形状：4096×4096 = 16.8M 元素，读 + 写 = 134.2 MB，理论下限 524.2 µs。
# 与第 3 章归约同量级，时间足够长，计时误差可以忽略。
BENCH_SHAPES = [(4096, 4096)]


def _variants() -> dict[str, str]:
    """一变体一符号：transpose_<变体>（只有一个算子，不再拼算子名）。"""
    return {label: f"transpose_{label}" for label in VARIANT_ORDER}


def _reference(x):
    """参考实现：float64 上转置。

    `transpose(0,1)` 本身是**视图**，`.contiguous()` 才把结果真正落成内存顺序 ——
    后者才是本仓库的 kernel 要产出的东西（一个 (C,R) 的连续张量）。
    """
    return x.transpose(0, 1).contiguous()


OPS = {
    "transpose": {
        "variants": _variants(),
        "reference": _reference,
        "shapes": EDGE_SHAPES,
        "bench_shapes": BENCH_SHAPES,
        # 输出形状 = (cols, rows)：这是 `out_shape` 字段存在的理由
        "out_shape": lambda shape: (shape[1], shape[0]),
        # 转置是纯数据搬运，输入分布不影响结果；用 randn 只是为了让
        # "搬错位置"产生的差异更容易看出来（正负都有）。
        "gen": lambda shape: (torch.randn(shape, dtype=torch.float32),),
        # 读 4MN + 写 4MN
        "bytes": lambda shape: 8 * shape[0] * shape[1],
        # 一个算术运算都没有：这一列恒为 0 正好说明"纯搬运"这个极端
        "flops": lambda shape: 0,
        # **精确相等**：转置是置换，没有任何舍入余地
        "rtol": 0.0,
        "atol": 0.0,
        "torch": _reference,
    },
}
