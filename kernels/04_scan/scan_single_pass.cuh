// kernels/04_scan/scan_single_pass.cuh
//
// 单趟 decoupled look-back 的 kernel 与 launcher，**模板参数 K = 每线程元素数**。
//
// v3 与 v5 是同一个 kernel 的两次实例化：
//
//     v3_single_pass   K = 1   chunk = 256   —— 不做摊销
//     v5_optimized     K = 8   chunk = 2048  —— 把 per-chunk 开销摊销 8 倍
//
// 把差异压成一个模板参数，是为了让"v3 → v5 到底改了什么"在 diff 里一眼可见。
//
// 块内算法两个都取 **Kogge-Stone**：v1/v2 已经实测出 KS 比 Blelloch 快 25%
// （Blelloch 级数 16 对 8，而且逐级变窄）。所以 v2 → v3 实际上同时改了两件事 ——
// 换多 block 合并方式、以及按实测结论把块内换成 KS。
// **这不是疏忽，而是阶梯的正确用法：后面的级应当建立在前面的结论上。**
// 变体的名字里也写明了这一点（见 scan.py 的 VARIANT_NOTES）。
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "kernels/04_scan/scan.cuh"
#include "kernels/04_scan/scan_chunked.cuh"
#include "kernels/04_scan/scan_lookback.cuh"
#include "kernels/common/cuda_check.h"

namespace ops_lab {
namespace scan {

template <int K>
__global__ void scan_single_pass_kernel(const float* __restrict__ x, float* __restrict__ out,
                                       int64_t n, int64_t chunks,
                                       float* __restrict__ aggregate,
                                       float* __restrict__ inclusive,
                                       int* __restrict__ status) {
  __shared__ float smem[chunked_scratch_floats<K>()];
  const int tid = static_cast<int>(threadIdx.x);
  constexpr int64_t kChunkElems = static_cast<int64_t>(kBlockSize) * K;

  for (int64_t c = blockIdx.x; c < chunks; c += gridDim.x) {
    const int64_t base = c * kChunkElems;
    float local[K];
    float t_exclusive = 0.0f;
    float total = 0.0f;

    // ① 载入 + 两级块内扫描（结果还在寄存器/smem 里，**先不写全局**）
    chunk_scan_two_level<KoggeStoneScanPolicy, K>(x, base, n, smem, local, t_exclusive, total);

    // ② 先把 chunk 总和发布出去，让后继 block 至少能靠它往前走
    if (tid == 0) {
      aggregate[c] = total;
      __threadfence();
      atomicExch(&status[c], kStatusAggregate);
    }

    // ③ warp 0 并行回看（整 warp 一次比 32 个 predecessor）
    if (tid < 32) {
      const float prefix = warp_lookback(c, aggregate, inclusive, status);
      if (tid == 0) {
        smem[chunked_prefix_slot<K>()] = prefix;
        // ④ **先发布 inclusive 前缀、再写回。** 写回是这条关键路径上最慢的一段，
        //    让它后面的 block 尽早停止回看，比"写完了再发布"快得多。
        inclusive[c] = prefix + total;
        __threadfence();
        atomicExch(&status[c], kStatusInclusive);
      }
    }
    __syncthreads();  // 等 warp 0 把前缀放进广播槽

    // ⑤ 加前缀 → 经 smem 转置回 striped → 合并写回
    chunk_store<K>(out, base, n, smem, local, t_exclusive, smem[chunked_prefix_slot<K>()]);
    __syncthreads();  // 下一轮要复用 smem
  }
}

// ---------------------------------------------------------------------------
// launcher
//
// workspace 布局：aggregate[chunks] / inclusive[chunks] / status[chunks]
// 其中 chunks 是**本 K 下**的 chunk 数。绑定层按 kChunkSize=256 的最坏情况
// （即 K=1）分配，所以对 K>1 只会多分配，不会越界 —— 这一点写在
// `workspace_floats()` 的注释里。
// ---------------------------------------------------------------------------
template <int K>
inline void launch_single_pass(const float* x, float* out, int64_t n, float* workspace,
                               cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  constexpr int64_t kChunkElems = static_cast<int64_t>(kBlockSize) * K;
  const int64_t chunks = (n + kChunkElems - 1) / kChunkElems;

  const int64_t cap = default_grid_size(kBlockSize);
  const int64_t grid = chunks < cap ? chunks : cap;

  // 死锁护栏：自旋等待要求所有 block 同时驻留（见 scan_lookback.cuh）
  require_co_resident(grid, n);

  if (workspace == nullptr) {
    throw std::runtime_error(
        "scan 单趟实现需要 workspace（aggregate / inclusive / status 三段），"
        "传 nullptr 是调用方的错误：\n  n = " + std::to_string(n) + "，chunks = " +
        std::to_string(chunks) + "（K = " + std::to_string(K) + "）→ 需要 " +
        std::to_string(3 * chunks) + " 个 float。");
  }

  float* aggregate = workspace;
  float* inclusive = workspace + chunks;
  int* status = reinterpret_cast<int*>(workspace + 2 * chunks);

  // status 必须全部清零（0 = 未发布）。这是唯一一处"每个 chunk 都要写一遍"的
  // 额外流量：K=1 时是 128 KB，占 8N 的 0.2%；K=8 时只有 16 KB。
  OPSLAB_CUDA_CHECK(
      cudaMemsetAsync(status, 0, sizeof(int) * static_cast<size_t>(chunks), stream));

  scan_single_pass_kernel<K><<<static_cast<unsigned int>(grid), kBlockSize, 0, stream>>>(
      x, out, n, chunks, aggregate, inclusive, status);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

}  // namespace scan
}  // namespace ops_lab
