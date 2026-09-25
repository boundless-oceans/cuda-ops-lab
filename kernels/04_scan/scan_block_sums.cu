// kernels/04_scan/scan_block_sums.cu
//
// 三段式的前两趟 —— v0~v2 共用的部分。
//
// 分成两趟是**scan 独有的代价**（reduction 没有这个问题）：要算第 c 个 chunk 的
// 前缀，必须先知道前面所有 chunk 的和；而"前面所有 chunk 的和"只能等它们都算完。
// 于是只能：
//
//     A 趟：读一遍 x，每个 chunk 缩成一个和
//     B 趟：把这 C 个和扫成前缀
//     C 趟：再读一遍 x，块内扫描 + 加上前缀，写回
//
// **读了两遍 x** —— 这就是 12N 的来源（理想是 8N）。
// 单趟实现（scan_v3_single_pass.cu）存在的全部意义就是把这一遍省掉，
// 代价是要用 decoupled look-back 在 block 之间建立顺序。
#include "kernels/04_scan/scan_kernels.cuh"

#include "kernels/common/cuda_check.h"

namespace ops_lab {
namespace scan {

// ------------------------------------------------------------------- A 趟
//
// grid = chunk 数，一个 block 负责一个 chunk。
// 块内用 smem 树归约（第 3 章那棵顺序寻址的树）。
__global__ void chunk_sums_kernel(const float* __restrict__ x, float* __restrict__ sums,
                                  int64_t n) {
  __shared__ float smem[kBlockSize + 1];

  const int64_t gi = static_cast<int64_t>(blockIdx.x) * kChunkSize + threadIdx.x;
  smem[threadIdx.x] = (gi < n) ? x[gi] : 0.0f;

  const float total = block_reduce_sum<kBlockSize>(smem);
  if (threadIdx.x == 0) {
    sums[blockIdx.x] = total;
  }
}

// ------------------------------------------------------------------- B 趟
//
// **只有一个 block**：C 个 chunk 和（C = n/256）用一个 block 扫完。
//
// 为什么可以这么奢侈：这一趟的数据量是 N/256，比 N 小两个数量级，用它换
// "不需要第二层递归"非常划算。真正大 N 的实现会把这一趟也拆成多 block
// （即递归的 scan-scan-add），本仓库不做 —— 那是同一件事的重复。
//
// 做法是每线程先**串行**扫自己那一段（保证顺序正确），再对线程间的
// 256 个总量做一次块内扫描，最后把线程内的前缀补回去。
__global__ void scan_chunk_sums_kernel(float* __restrict__ sums, int64_t count) {
  __shared__ float smem[kBlockSize + 1];

  const int t = static_cast<int>(threadIdx.x);
  const int64_t per_thread = (count + kBlockSize - 1) / kBlockSize;
  const int64_t lo = static_cast<int64_t>(t) * per_thread;
  const int64_t hi = (lo + per_thread < count) ? (lo + per_thread) : count;

  float t_sum = 0.0f;
  for (int64_t i = lo; i < hi; ++i) {
    t_sum += sums[i];
  }
  smem[t] = t_sum;

  // 复用 Kogge-Stone：smem[t] 出来的就是"前面所有线程的和"
  kogge_stone_block_scan_exclusive<kBlockSize>(smem, t_sum);
  const float t_exclusive = smem[t];

  float running = t_exclusive;
  for (int64_t i = lo; i < hi; ++i) {
    const float v = sums[i];
    sums[i] = running;
    running += v;
  }
}

}  // namespace scan
}  // namespace ops_lab
