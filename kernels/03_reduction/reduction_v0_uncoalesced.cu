// kernels/03_reduction/reduction_v0_uncoalesced.cu
//
// 第 3 章 · 阶梯第 1 级：反面教材 —— 访存未合并。
//
// 与 v1_naive 相比**只有索引公式那一行不同**（见下面标了 ★ 的地方），
// 其余（grid 计算、grid-stride 循环、块内 smem 树、每 block 一次原子合并）
// 完全一样。这样阶梯表里 v0 → v1 的差距就只能归因于访存模式。
#include "kernels/03_reduction/reduction.cuh"
#include "kernels/03_reduction/reduction_block.cuh"

#include "kernels/common/cuda_check.h"

namespace ops_lab {
namespace reduction {
namespace {

template <typename Op>
__global__ void reduce_v0_uncoalesced_kernel(const float* __restrict__ x,
                                             float* __restrict__ out, int64_t n) {
  float acc = Op::kIdentity;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  // ★ 唯一与 v1 不同的地方：两个乘法项换了位置。
  //   相邻的 threadIdx 之间因此相隔 gridDim.x 个元素（本机 144 个 = 576 字节），
  //   一个 warp 的 32 个线程落在 32 条不同的 cache line 上。
  //
  //   仍然正确的理由（这一条必须自己想清楚，否则会写出丢元素的版本）：
  //   (threadIdx.x, blockIdx.x) → threadIdx.x * gridDim.x + blockIdx.x 是
  //   [0, blockDim.x * gridDim.x) 上的一一映射，而循环步长恰好等于这个区间长度，
  //   所以每个下标被恰好一个线程访问一次。
  for (int64_t i = static_cast<int64_t>(threadIdx.x) * gridDim.x + blockIdx.x; i < n;
       i += stride) {
    acc = Op::combine(acc, x[i]);
  }

  const float block_total = block_reduce_smem_tree<Op>(acc);
  if (threadIdx.x == 0) {
    Op::atomic_combine(out, block_total);
  }
}

}  // namespace

void sum_v0_uncoalesced(const float* x, float* out, int64_t n, cudaStream_t stream) {
  init_reduction_output(out, SumOp::kIdentity, stream);
  if (n <= 0) {
    return;
  }
  reduce_v0_uncoalesced_kernel<SumOp><<<grid_for_reduction(n), kBlockSize, 0, stream>>>(x, out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

void max_v0_uncoalesced(const float* x, float* out, int64_t n, cudaStream_t stream) {
  init_reduction_output(out, MaxOp::kIdentity, stream);
  if (n <= 0) {
    return;
  }
  reduce_v0_uncoalesced_kernel<MaxOp><<<grid_for_reduction(n), kBlockSize, 0, stream>>>(x, out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

}  // namespace reduction
}  // namespace ops_lab
