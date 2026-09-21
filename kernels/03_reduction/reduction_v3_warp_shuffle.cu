// kernels/03_reduction/reduction_v3_warp_shuffle.cu
//
// 第 3 章 · 阶梯第 4 级：块内合并改用 warp shuffle。
//
// 与 v1_naive 相比**只差块内合并调用的是哪个函数**
// （block_reduce_smem_tree → block_reduce_warp_shuffle）。
// 共享内存访问从 8×256 次降到 8 次，__syncthreads 从 8 次降到 1 次。
//
// 预期**≈ v1**：按 reduction.cuh 里那笔账，8 级同步总共约 0.25 µs，
// 而 kernel 本身是 262 µs 量级 —— 省下的比测量噪声还小。
#include "kernels/03_reduction/reduction.cuh"
#include "kernels/03_reduction/reduction_block.cuh"

#include "kernels/common/cuda_check.h"

namespace ops_lab {
namespace reduction {
namespace {

template <typename Op>
__global__ void reduce_v3_warp_shuffle_kernel(const float* __restrict__ x,
                                              float* __restrict__ out, int64_t n) {
  float acc = Op::kIdentity;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    acc = Op::combine(acc, x[i]);
  }

  // 与 v1 的唯一差别：块内合并走 shuffle，不用 smem 树
  const float block_total = block_reduce_warp_shuffle<Op>(acc);

  if (threadIdx.x == 0) {
    Op::atomic_combine(out, block_total);
  }
}

}  // namespace

void sum_v3_warp_shuffle(const float* x, float* out, int64_t n, cudaStream_t stream) {
  init_reduction_output(out, SumOp::kIdentity, stream);
  if (n <= 0) {
    return;
  }
  reduce_v3_warp_shuffle_kernel<SumOp><<<grid_for_reduction(n), kBlockSize, 0, stream>>>(x, out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

void max_v3_warp_shuffle(const float* x, float* out, int64_t n, cudaStream_t stream) {
  init_reduction_output(out, MaxOp::kIdentity, stream);
  if (n <= 0) {
    return;
  }
  reduce_v3_warp_shuffle_kernel<MaxOp><<<grid_for_reduction(n), kBlockSize, 0, stream>>>(x, out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

}  // namespace reduction
}  // namespace ops_lab
