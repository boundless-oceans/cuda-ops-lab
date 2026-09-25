"""第 4 章：scan（inclusive prefix sum 的五级阶梯）。

## 这一章的元数据多了两个新东西

1. **逐变体的 `bytes`**：三段式（v0~v2）**读两遍输入**，所以全局流量是 12N；
   单趟实现（v3/v4）只读一遍，是 8N。用同一个 `bytes` 去算 `%峰值`，
   等于拿两把尺子量东西 —— 所以 `variants` 声明里可以直接覆盖 `bytes`
   （见 `registry.variant_bytes()`），表里同时给出"实际搬运"和"按各自实际流量
   算的 %峰值"。

   于是这一章会读出一个前三章没机会出现的结论：
   **`%峰值` 高不等于快。** v1/v2 把自己的 12N 用到 93%，仍然比把 8N 用到 95%
   的 v3 慢 45%。

2. **`workspace` 第一次真的用上了**：单趟实现需要 aggregate / inclusive / status
   三段（为什么必须分开存，见 `kernels/04_scan/scan_config.cuh`）。
"""

from __future__ import annotations

import torch

NAME = "04_scan"

VARIANT_ORDER = (
    "v0_naive",
    "v1_kogge_stone",
    "v2_blelloch",
    "v3_single_pass",
    "v4_optimized",
    "v5_single_block",
)

VARIANT_NOTES = {
    "v0_naive": "反面教材：三段式 + 块内朴素扫描（每元素 O(B)，work 远大于内存时间）",
    "v1_kogge_stone": "三段式 + 块内 Kogge-Stone（8 级，O(N log B)）；12N 流量",
    "v2_blelloch": "三段式 + 块内 Blelloch 上扫/下扫（work 最少但级数 16）；仍 12N 流量",
    "v3_single_pass": "单趟 look-back（K=1）：只读一遍输入（8N），但每 chunk 开销没摊销",
    "v4_optimized": "单趟 look-back + 每线程 8 个元素（chunk 2048）：摊销 per-chunk 开销",
    "v5_single_block": "故意回退：grid = 1，一个 block 顺序走完所有 chunk（8N 也没用）",
}

# 形状必须覆盖这些情况（scan 的边界比归约更密，因为"每个 chunk 的边界"都会
# 影响前缀，而前缀错了会往后传染整段）：
#   0            空输入（输出也是空的，绑定层直接返回）
#   1            只有一个元素
#   31/32/33     一个 warp 的三个位置
#   255/256/257  一个 chunk 的三个位置（尾部补齐 + 前缀跨 chunk 传递）
#   511/512/513  两个 chunk 的三个位置（跨 chunk 边界最容易错）
#   1024         多个 chunk
#   1000003      大、奇数、非 2 的幂
EDGE_SHAPES = [(0,), (1,), (31,), (32,), (33,), (255,), (256,), (257,), (511,), (512,), (513,),
               (1024,), (1000003,)]

# 基准专用形状。
#
# N = 2²³ 是**刻意比第 2、3 章小一档**的：v0 的块内 work 是 O(N·B)，
# N = 2²⁴ 时它单次要跑约 2.8 ms（3 轮 × 60 次 = 几十秒）。
# 2²³ 时 8N 的搬运量正好是 67.1 MB，理论下限 262.1 µs —— 与第 3 章同量级，
# 横向对比仍然成立。
BENCH_SHAPES = [(1 << 23,)]

# 与 kernels/04_scan/scan_config.cuh 的 kChunkSize 保持一致。
# 只用来估算 workspace 的那点流量（占比 <0.5%），不同步也不会误报。
CHUNK = 256

# 与 kernels/04_scan/scan_config.cuh 的 kOptimizedItemsPerThread 保持一致。
OPTIMIZED_ITEMS_PER_THREAD = 8


def _chunks(shape, items_per_thread: int = 1) -> int:
    """该形状下的 chunk 数。K 越大 chunk 越大、chunk 数越少。"""
    chunk = CHUNK * items_per_thread
    n = shape[0]
    return 0 if n <= 0 else -(-n // chunk)


def _variants() -> dict[str, dict]:
    """一变体一符号，外加**各自的** bytes（见模块开头第 1 点）。"""
    return {
        # 三段式：A 趟读 x + B 趟扫 chunk 和 + C 趟再读 x 并写 out
        #   主流量 = 读 4N + 读 4N + 写 4N = 12N
        #   workspace 流量 ≈ 4 遍 C 个 float（A 写 / B 读+写 / C 读）= 16C 字节
        "v0_naive": {
            "symbol": "scan_inclusive_v0_naive",
            "bytes": lambda shape: 12 * shape[0] + 16 * _chunks(shape, 1),
        },
        "v1_kogge_stone": {
            "symbol": "scan_inclusive_v1_kogge_stone",
            "bytes": lambda shape: 12 * shape[0] + 16 * _chunks(shape, 1),
        },
        "v2_blelloch": {
            "symbol": "scan_inclusive_v2_blelloch",
            "bytes": lambda shape: 12 * shape[0] + 16 * _chunks(shape, 1),
        },
        # 单趟：读 4N + 写 4N = 8N
        #   workspace 流量 ≈ 6 遍 C（status 清零 / aggregate 写+读 / inclusive 写+读）= 24C 字节
        #   注意 v3 是 K=1（chunk 256）、v4 是 K=8（chunk 2048），
        #   workspace 的项因此差 8 倍 —— 但两者都只占 8N 的 <0.3%，不影响结论
        "v3_single_pass": {
            "symbol": "scan_inclusive_v3_single_pass",
            "bytes": lambda shape: 8 * shape[0] + 24 * _chunks(shape, 1),
        },
        "v4_optimized": {
            "symbol": "scan_inclusive_v4_optimized",
            "bytes": lambda shape: 8 * shape[0] + 24 * _chunks(shape, OPTIMIZED_ITEMS_PER_THREAD),
        },
        # 单 block：读 4N + 写 4N，不需要 workspace
        "v5_single_block": {
            "symbol": "scan_inclusive_v5_single_block",
            "bytes": lambda shape: 8 * shape[0],
        },
    }


def _reference(x):
    """参考实现：float64 上做前缀和。

    **多维输入按展平处理**（kernel 只认 numel 和连续内存，与第 2、3 章一致）。
    所以形状要 reshape 回去，而不是按最后一维扫 —— 后者是另一类算子
    （逐行 softmax/norm 那一路），第 6、7 章才讲。
    """
    return x.reshape(-1).cumsum(0).reshape(x.shape)


OPS = {
    "inclusive_scan": {
        "variants": _variants(),
        "reference": _reference,
        "shapes": EDGE_SHAPES,
        "bench_shapes": BENCH_SHAPES,
        # 全正数输入：前缀和单调增长，相对误差才有意义（正负相消会让它失去意义，
        # 与第 3 章的 sum 是同一个理由）。
        "gen": lambda shape: (torch.rand(shape, dtype=torch.float32),),
        # 算子级 bytes = **理想**流量（8N）。它是表头里"理论最短耗时"的来源，
        # 也是"实际搬运"那一列的比较基准；各变体真正的 bytes 在上面各自声明。
        "bytes": lambda shape: 8 * shape[0],
        # 这里的 flops 是**理想算法**的加法次数（每个输出一个加法），只用来判断
        # "算术强度 vs machine balance"，结论是 memory-bound。
        #
        # 注意：各变体的**真实**加法次数差得很远 —— 朴素扫描是 N·B 量级
        # （每元素 O(B)），Kogge-Stone 是 N log B，Blelloch 是 2N。
        # 这个差异正是第 4 章的主线，所以它不能塞进一个数里：
        # 表里的 median 才是真的，flops 这一列只是类型判断用的名义值。
        "flops": lambda shape: max(shape[0] - 1, 0),
        # 前缀和是累加，误差随前缀长度增长（量级同步增长，所以相对误差稳定）。
        # 取 1e-3 是留给"两级结构 + 串行段"的余量；**真正锋利的检查在边界测试里**：
        # 全 1 输入时前缀和恰好是 i+1，float32 可以精确表示，所以那里要求逐位相等。
        "rtol": 1e-3,
        "atol": 1e-4,
        "torch": _reference,
    },
}
