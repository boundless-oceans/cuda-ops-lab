"""第 2 章：elementwise（add / relu / sigmoid 的五级优化阶梯）。

三个算子的 kernel 是共用模板（见 `kernels/02_elementwise/elementwise_ops.cuh`），
所以这里也用循环生成变体表，而不是手写 15 行 —— 少一处手抄就少一处不同步。
"""

from __future__ import annotations

import torch

NAME = "02_elementwise"

# 阶梯顺序。基准脚本按这个顺序排列阶梯表；字典序恰好也是这个顺序
# （v0 < v1 < v2 < v3 < v4），但显式写出来更不容易被后来的命名打乱。
VARIANT_ORDER = (
    "v0_uncoalesced",
    "v1_naive",
    "v2_unroll4",
    "v3_grid_stride",
    "v4_vectorized",
)

VARIANT_NOTES = {
    "v0_uncoalesced": "反面教材：相邻线程地址相隔 gridDim.x，一个 warp 跨 32 条 cache line",
    "v1_naive": "一线程一元素、连续访问（已合并访存）；无循环，grid 有上限",
    "v2_unroll4": "每线程 4 个元素：先全部 load 再统一 store，用 ILP 换并发度",
    "v3_grid_stride": "grid 固定成“刚好填满机器” + 循环步进，解除 grid 上限",
    "v4_vectorized": "float4，访存指令数降 4 倍；非 4 倍数走标量尾部",
}

# 形状必须覆盖这些情况，否则边界 bug 测不出来：
#   0        空输入（绑定层必须直接返回空张量，不能启动 grid==0 的 kernel）
#   1 / 3    极小；且 1 和 3 不是 4 的倍数，专门打 v4 的标量尾巴
#   7        非 2 的幂，且不是 4 的倍数
#   255/256/257  一个 block 的三个位置：不满 / 恰好 / 多一个
#   1024     多个 block，且是 4 的倍数（v4 走纯向量路径）
#   1000003  大、奇数、非 2 的幂 —— 三个不利条件叠在一起
EDGE_SHAPES = [(0,), (1,), (3,), (7,), (255,), (256,), (257,), (1024,), (1000003,)]


def _variants(op: str) -> dict[str, str]:
    """一变体一符号：elementwise_<算子>_<变体>。"""
    return {label: f"elementwise_{op}_{label}" for label in VARIANT_ORDER}


OPS = {
    "add": {
        "variants": _variants("add"),
        "reference": lambda x, y: x + y,
        "shapes": EDGE_SHAPES,
        # 读 x（4N）+ 读 y（4N）+ 写 out（4N）
        "bytes": lambda shape: shape[0] * 4 * 3,
        "flops": lambda shape: shape[0],
        "torch": lambda x, y: x + y,
    },
    "relu": {
        "variants": _variants("relu"),
        "reference": lambda x: torch.clamp_min(x, 0.0),
        "shapes": EDGE_SHAPES,
        "bytes": lambda shape: shape[0] * 4 * 2,
        "flops": lambda shape: shape[0],
        "torch": lambda x: torch.relu(x),
    },
    "sigmoid": {
        "variants": _variants("sigmoid"),
        "reference": lambda x: torch.sigmoid(x),
        "shapes": EDGE_SHAPES,
        "bytes": lambda shape: shape[0] * 4 * 2,
        # exp 的运算次数不计入：它既不是一个 FMA，也不是本算子的知识重点。
        # 这一列只用来判断"算术强度 vs machine balance"，见性能分析文档第 0 步。
        "flops": lambda shape: shape[0],
        "torch": lambda x: torch.sigmoid(x),
    },
}
