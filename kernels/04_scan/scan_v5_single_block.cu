// kernels/04_scan/scan_v5_single_block.cu
//
// 第 4 章 · 阶梯第 6 级：**故意回退** —— 整个数组交给一个 block。
//
// 块内沿用本章最省事的那套（Blelloch，每线程一个元素），但 grid = 1：
// 一个 block 顺序走过所有 chunk，用寄存器里的一个 carry 把 chunk 之间串起来。
//
// 放在阶梯最后一级是刻意的：它对比的对象是**已经调到位的 v4**，而不是从零开始的
// v0。这样表上读到的就是"结构、块内算法、摊销都选对之后，只用一个 block 会怎样"。
//
// 实测（第一版，chunk=256/Blelloch）：**31098.9 µs**，0.8% 峰值。
// 也就是说 32768 次 chunk 扫描 × 大约 18 次 barrier 全部串行在**一个 SM** 上 ——
// 31 ms ÷ 32768 ≈ 0.95 µs/chunk，正好是 18 次 barrier 的量级。
//
// 和第 3 章的 v4_single_block（慢 16.3 倍）是同一个教训：
// **用不满机器是第一位的问题，其余都在它后面。**
//
// 不需要 workspace：carry 在寄存器里，chunk 之间是纯串行依赖。
#include "kernels/04_scan/scan.cuh"

#include "kernels/04_scan/scan_kernels.cuh"
#include "kernels/common/cuda_check.h"

namespace ops_lab {
namespace scan {
namespace {

__global__ void scan_single_block_kernel(const float* __restrict__ x, float* __restrict__ out,
                                        int64_t n, int64_t chunks) {
  __shared__ float smem[kBlockSize + 1];
  const int tid = static_cast<int>(threadIdx.x);

  // 所有线程都维护同一个 carry（每轮加的是同一个 total），所以它天然保持一致
  float carry = 0.0f;
  for (int64_t c = 0; c < chunks; ++c) {
    const int64_t base = c * kChunkSize;
    const int64_t gi = base + tid;
    const float v = (gi < n) ? x[gi] : 0.0f;
    smem[tid] = v;

    const float total = blelloch_block_scan_exclusive<kBlockSize>(smem, v);

    if (gi < n) {
      out[gi] = smem[tid] + v + carry;
    }
    carry += total;
    __syncthreads();  // 下一轮会覆盖 smem，等所有线程读完（保险，也表明意图）
  }
}

}  // namespace

void scan_inclusive_v5_single_block(const float* x, float* out, int64_t n, cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  scan_single_block_kernel<<<1, kBlockSize, 0, stream>>>(x, out, n, chunk_count(n));
  OPSLAB_CUDA_CHECK_LAUNCH();
}

}  // namespace scan
}  // namespace ops_lab
