// kernels/03_reduction/reduction_v4_single_block.cu
//
// 第 3 章 · 阶梯第 5 级：**故意回退** —— 只用一个 block。
//
// 块内沿用 v3 的 warp shuffle（本章最激进的那套块内写法），但启动时 grid = 1：
// 整个 2^24 的数组交给一个 block 的 256 个线程去读。
//
// 与 v3 的 diff 只有一行：`grid_for_reduction(n)` → `kSingleBlockGrid`。
// 这个变体存在的意义就是把"block 数"这一个变量单独拎出来量化 ——
// 前面 v1→v2→v3 折腾块内写法省下的量级是 0.1%，而这里少写一个 grid 参数
// 的代价是 10 倍以上。两者的对比就是本章对比赛最有价值的一句话。
//
// 预期 5~10% 峰值（约 2.6~5.2 ms）。
#include "kernels/03_reduction/reduction.cuh"
#include "kernels/03_reduction/reduction_block.cuh"

#include "kernels/common/cuda_check.h"

namespace ops_lab {
namespace reduction {
namespace {

// 故意的：整个 GPU 只用一个 block
constexpr unsigned int kSingleBlockGrid = 1;

template <typename Op>
__global__ void reduce_v4_single_block_kernel(const float* __restrict__ x,
                                              float* __restrict__ out, int64_t n) {
  float acc = Op::kIdentity;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    acc = Op::combine(acc, x[i]);
  }

  const float block_total = block_reduce_warp_shuffle<Op>(acc);

  if (threadIdx.x == 0) {
    Op::atomic_combine(out, block_total);
  }
}

}  // namespace

void sum_v4_single_block(const float* x, float* out, int64_t n, cudaStream_t stream) {
  init_reduction_output(out, SumOp::kIdentity, stream);
  if (n <= 0) {
    return;
  }
  reduce_v4_single_block_kernel<SumOp><<<kSingleBlockGrid, kBlockSize, 0, stream>>>(x, out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

void max_v4_single_block(const float* x, float* out, int64_t n, cudaStream_t stream) {
  init_reduction_output(out, MaxOp::kIdentity, stream);
  if (n <= 0) {
    return;
  }
  reduce_v4_single_block_kernel<MaxOp><<<kSingleBlockGrid, kBlockSize, 0, stream>>>(x, out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

}  // namespace reduction
}  // namespace ops_lab
