"""第 1 章：执行模型（两个受控实验）。

这一章不是"优化阶梯"，所以它的算子和第 2 章的性质不同：

* `hello` 的输入是**一个整数 n**，不是张量；
* `bandwidth_probe` 是**一个符号配一个 variant 参数**（1..4 选择共享内存大小），
  不是"一变体一符号"。

这两点正好用上了注册表支持的两种变体声明写法，也说明元数据结构不能假设
"所有算子都长得一样"。
"""

from __future__ import annotations

import torch

NAME = "01_execution"

VARIANT_NOTES = {
    "v1_naive": "一线程一元素、精确 grid、带边界检查；grid 有上限（2^31-1）",
    "v2_grid_stride": "grid 固定、kernel 内循环步进；无 grid 上限，也消除尾部空转",
    "probe_smem0": "occupancy 实验：动态共享内存 0 KB（占用率最高）",
    "probe_smem24k": "occupancy 实验：动态共享内存 24 KB",
    "probe_smem32k": "occupancy 实验：动态共享内存 32 KB",
    "probe_smem48k": "occupancy 实验：动态共享内存 48 KB（占用率最低）",
}

OPS = {
    # ------------------------------------------------------------------
    # 实验一：hello —— grid/block/thread 的心智模型
    # ------------------------------------------------------------------
    "hello": {
        "variants": {
            "v1_naive": "execution_hello_v1_naive",
            # v2 比 v1 多两个参数：block_size 与 grid_size（grid_size=0 表示
            # 按设备规模自动选）。这里显式给 (256, 0)，而不是改成 binding 的
            # 默认参数 —— 一是注册表本来就支持"一个符号 + 额外参数"，二是
            # 第 1 章本来就是关于 grid 的实验，把 block/grid 显式写出来更诚实。
            "v2_grid_stride": {
                "symbol": "execution_hello_v2_grid_stride",
                "args": [256, 0],
            },
        },
        # 输入不是张量，而是一个整数 n —— 所以必须自定义 gen
        "gen": lambda shape: (shape[0],),
        "reference": lambda n: torch.arange(n, dtype=torch.int32),
        # n=0（空）、n=1（极小）、n=1024（常规）、n=2^20（大，且 v1 的 grid 还装得下）
        "shapes": [(0,), (1,), (1024,), (1 << 20,)],
        # 基准形状必须**远大于 L2**，否则测的是缓存带宽而不是显存带宽。
        # hello 只写 4N 字节，2^24 时输出 64 MB，够用。
        "bench_shapes": [(1 << 24,)],
        # 纯写：只写 out[i] = i，输出 4 字节/元素
        "bytes": lambda shape: 4 * shape[0],
        "flops": lambda shape: shape[0],
        # 整数索引，必须逐元素**精确相等**，不能有容差
        "rtol": 0.0,
        "atol": 0.0,
    },
    # ------------------------------------------------------------------
    # 实验二：bandwidth_probe —— occupancy 受控实验
    # ------------------------------------------------------------------
    # 四个变体是同一个符号，靠额外参数选择共享内存大小。
    "bandwidth_probe": {
        "variants": {
            "probe_smem0": {"symbol": "execution_bandwidth_probe", "args": [1]},
            "probe_smem24k": {"symbol": "execution_bandwidth_probe", "args": [2]},
            "probe_smem32k": {"symbol": "execution_bandwidth_probe", "args": [3]},
            "probe_smem48k": {"symbol": "execution_bandwidth_probe", "args": [4]},
        },
        "reference": lambda x: x.clone(),
        # n=4 时 n4=1，只有 1 个 float4；n=1 时 n4=0，全靠标量尾巴
        "shapes": [(0,), (1,), (4,), (1024,), (1 << 20,)],
        # 纯拷贝：读 4N + 写 4N。2^24 时工作集 134 MB，远超 L2
        "bench_shapes": [(1 << 24,)],
        # 纯拷贝：读 4N + 写 4N
        "bytes": lambda shape: 4 * shape[0] * 2,
        "flops": lambda shape: 0,
        # 纯拷贝，必须逐位相等
        "rtol": 0.0,
        "atol": 0.0,
    },
}

# 同样是导出的符号，但不是算子、也没有"变体"概念 ——
# 一致性校验会跳过它们，但仍然会校验它们确实被导出了。
UTILITIES = ("execution_probe_occupancy",)
