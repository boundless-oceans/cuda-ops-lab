// kernels/03_reduction/reduction_v2_smem_tree.cu
//
// 第 3 章 · 阶梯第 3 级：给共享内存加 padding，消 bank conflict。
//
// 与 v1_naive 相比**只差块内合并调用的是哪个函数**
// （block_reduce_smem_tree → block_reduce_smem_tree_padded，见 reduction_block.cuh），
// 读元素和最终原子合并是同一份代码。
//
// 这一级的预期是**测不出差别** —— 因为 v1 用的顺序寻址本来就没有 bank conflict。
// 详见 reduction_block.cuh 里 v1/v2 两段注释，以及 reduction.cuh 里"一个必须先说
// 清楚的预测"那一节。
#include "kernels/03_reduction/reduction.cuh"
#include "kernels/03_reduction/reduction_block.cuh"

#include "kernels/common/cuda_check.h"

namespace ops_lab {
namespace reduction {
namespace {

template <typename Op>
__global__ void reduce_v2_smem_tree_kernel(const float* __restrict__ x, float* __restrict__ out,
                                           int64_t n) {
  float acc = Op::kIdentity;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    acc = Op::combine(acc, x[i]);
  }

  // 与 v1 的唯一差别：带 padding 的 smem 布局
  const float block_total = block_reduce_smem_tree_padded<Op>(acc);

  if (threadIdx.x == 0) {
    Op::atomic_combine(out, block_total);
  }
}

}  // namespace

void sum_v2_smem_tree(const float* x, float* out, int64_t n, cudaStream_t stream) {
  init_reduction_output(out, SumOp::kIdentity, stream);
  if (n <= 0) {
    return;
  }
  reduce_v2_smem_tree_kernel<SumOp><<<grid_for_reduction(n), kBlockSize, 0, stream>>>(x, out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

void max_v2_smem_tree(const float* x, float* out, int64_t n, cudaStream_t stream) {
  init_reduction_output(out, MaxOp::kIdentity, stream);
  if (n <= 0) {
    return;
  }
  reduce_v2_smem_tree_kernel<MaxOp><<<grid_for_reduction(n), kBlockSize, 0, stream>>>(x, out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

}  // namespace reduction
}  // namespace ops_lab
