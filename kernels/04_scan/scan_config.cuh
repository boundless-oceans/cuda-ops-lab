// kernels/04_scan/scan_config.cuh
//
// 第 4 章 scan 的启动配置与 workspace 布局。**只有常量，没有 device 代码。**
//
// 为什么单独拆出来：`bindings/bind_04_scan.cpp` 是宿主编译单元，它 include
// `scan.cuh` 时不能连带把 `__device__` 函数也拖进来 —— 这个坑第 3 章踩过
// （见 docs/DESIGN.md §5.1）。
#pragma once

#include <cstdint>

namespace ops_lab {
namespace scan {

// 块大小。256 = 8 个 warp：Kogge-Stone 8 级、Blelloch 上扫 8 级 + 下扫 8 级。
constexpr int kBlockSize = 256;
static_assert(kBlockSize % 32 == 0, "kBlockSize 必须是 32 的整数倍");

// 第 4 章的 N 是**按 chunk 数**分块的：一个 chunk = 一个 block 处理的一段。
constexpr int kChunkSize = kBlockSize;

// ---------------------------------------------------------------------------
// 两级块内扫描的"每线程元素数"（v5_optimized 用）
//
// 为什么要摊销：一个 chunk 要付 8 次 `__syncthreads()`，chunk 越大摊到每个元素上
// 越薄。每元素 barrier 数从 K=1 的 0.031 降到 K=8 的 0.0039。
//
// 为什么取 8 而不是 16：smem 需要 kBlockSize*(K+1)+1 个 float
//   K=8  → 2305 float = 9.2 KB → 每 SM 能放 6 个 block（受线程数限制，100% occupancy）
//   K=16 → 4353 float = 17.4 KB → 只能放 5 个 block（occupancy 掉到 83%）
// 这是一个**可以单独调的旋钮**：改这一个数不需要动任何算法。
constexpr int kOptimizedItemsPerThread = 8;

// 向上取整除法。scan 到处都要算 chunk 数，写一次。
inline int64_t chunk_count(int64_t n) {
  return n <= 0 ? 0 : (n + kChunkSize - 1) / kChunkSize;
}

// ---------------------------------------------------------------------------
// workspace 布局
//
// v0~v2（三段式）只需要 `block_prefix[chunk]` 一个数组；
// v3（单趟 look-back）需要三个：
//
//     aggregate[c]   第 c 个 chunk 的**总和**（先发布，供后继 look-back 累加）
//     inclusive[c]   第 c 个 chunk 的**inclusive 前缀**（后来发布，让后继提前停止）
//     status[c]      0 = 未发布，1 = 只有 aggregate，2 = 有 inclusive
//
// **为什么 aggregate 与 inclusive 必须分开存**：如果共用一个位置，会出现
// 这样的竞态 —— 读者读到 status = 「只有 aggregate」，然后去读 value，
// 而此时写者已经把 value 覆盖成 inclusive 前缀；读者按 aggregate 累加，
// 就会**把前面所有 chunk 重复算一遍**。这类 bug 只在时序恰好时出现，
// 所以宁可多花一个数组也不要共用。
//
// `status` 是 int，但 workspace 统一按 float 传（绑定层用 torch 分配），
// 所以 launcher 内部会把它 reinterpret 成 int*。
constexpr int kWorkspaceArrays = 3;

inline int64_t workspace_floats(int64_t n) {
  return kWorkspaceArrays * chunk_count(n);
}

// workspace 三段的偏移（单位：float）。参数只为调用处对称，暂时用不到。
inline int64_t aggregate_offset(int64_t /*n*/) { return 0; }
inline int64_t inclusive_offset(int64_t n) { return chunk_count(n); }
inline int64_t status_offset(int64_t n) { return 2 * chunk_count(n); }

}  // namespace scan
}  // namespace ops_lab
