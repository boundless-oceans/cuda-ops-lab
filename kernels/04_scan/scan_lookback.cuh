// kernels/04_scan/scan_lookback.cuh
//
// 单趟 scan 的跨 block 顺序建立：**整 warp 并行的** decoupled look-back。
//
// ## 为什么必须并行（第一版实测的教训）
//
// 第一版只让 `thread 0` 逐个往前找 predecessor。同一轮里 144 个 block 处理的是
// **相邻的 chunk**，所以回看距离最长可达 144 步 —— 每一步都是一次全局内存往返。
// 实测把 v3 拖到 **1874.9 µs**，而它的内存下限只有 262 µs（慢 7 倍）。
//
// 整 warp 一次比 32 个 predecessor，往返次数立刻除以 32。这不是"优化"，
// 而是这个算法的**标准形态**：少了它，decoupled look-back 根本不可用。
//
// ## 两个必须守住的细节
//
// ① **不能 per-lane 自旋。** 如果每个 lane 各自 `while (status == invalid) {}`，
//    同一个 warp 里的 lane 会分道扬镳，后面那个 `__ballot_sync` 就永远等不齐
//    —— 结果不是变慢，是**挂死**。所以重试必须是"整 warp 一起重试"：
//    先各读各的、再 ballot 看有没有没就绪的、有就整体重来。
//
// ② **状态与数据之间要有 `__threadfence()`。** 弱内存模型下，"看到状态"和
//    "看到状态对应的数据"是两件事。发布方先写数据、栅栏、再写状态；
//    读取方先读状态、栅栏、再读数据。
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ops_lab {
namespace scan {

// 发布状态。0 = 还没发布（由 launcher 的 memset 建立），所以取值都不能是 0。
constexpr int kStatusInvalid = 0;
constexpr int kStatusAggregate = 1;  // 只有 chunk 总和可用
constexpr int kStatusInclusive = 2;  // inclusive 前缀也可用（后继可以停止回看）

// 整 warp 并行回看。**只有 lane 0 的返回值有意义**，其余 lane 的返回值无定义。
//
// 调用方式：`if (threadIdx.x < 32) { prefix = warp_lookback(...); }`
// —— 32 个 lane 必须全部参与（里面用了 `__ballot_sync` / `__shfl_*_sync`）。
__device__ __forceinline__ float warp_lookback(int64_t c, const float* __restrict__ aggregate,
                                               const float* __restrict__ inclusive,
                                               int* __restrict__ status) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (c == 0) {
    return 0.0f;  // 第一个 chunk 没有前缀
  }

  float acc = 0.0f;
  int64_t window = c - 1;  // 本窗口里 lane 0 负责的 chunk 序号；lane 越大越靠前

  while (true) {
    const int64_t look = window - lane;
    int st = kStatusInvalid;

    // 整 warp 一起重试（见文件头 ①）。look < 0 的 lane 直接当作"到头了"：
    // 它们的 v 是 0，且因为 lane 序是递减的 chunk 序，它们一定是**最高**的几个 lane，
    // 所以不会挡住真正需要累加的 lane。
    while (true) {
      st = (look >= 0) ? atomicAdd(&status[look], 0) : kStatusInclusive;
      const unsigned not_ready = __ballot_sync(0xffffffffu, st == kStatusInvalid);
      if (not_ready == 0u) {
        break;
      }
    }
    __threadfence();  // 状态确认之后再读数据（见文件头 ②）

    float v = 0.0f;
    if (look >= 0) {
      v = (st == kStatusInclusive) ? inclusive[look] : aggregate[look];
    }

    // 本窗口里第一个"已经知道 inclusive 前缀"的 lane —— 它左边（更近的 predecessor）
    // 的账已经被它包含进去了，所以累加到它就可以停。
    const unsigned incl_mask = __ballot_sync(0xffffffffu, st == kStatusInclusive);
    const int first = incl_mask ? (__ffs(incl_mask) - 1) : 32;

    // 在 lane 序上做 inclusive 前缀和
    float x = v;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const float y = __shfl_up_sync(0xffffffffu, x, off);
      if (lane >= off) {
        x += y;
      }
    }
    const int last = (first < 32) ? first : 31;
    acc += __shfl_sync(0xffffffffu, x, last);

    if (first < 32) {
      break;  // 命中，pipeline 结束
    }
    window -= 32;
    if (window < 0) {
      break;  // 安全网：chunk 0 一定会发布 inclusive，正常情况下到不了这里
    }
  }
  return acc;
}

}  // namespace scan
}  // namespace ops_lab
