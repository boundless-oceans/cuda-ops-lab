// kernels/03_reduction/reduction_v1_naive.cu
//
// 第 3 章 · 阶梯第 2 级：结构正确但不做任何微调 —— 本章的**基准点**，
// 按预测也是本章的**答案**（~90% 峰值带宽）。
//
// 后面三级都从这里出发，各改一个地方：
//
//   v1 → v2  把 smem 下标从 tid 改成 tid + tid/32（消 bank conflict）
//   v1 → v3  块内合并换成 warp shuffle（同步 8 次 → 1 次）
//   v1 → v4  只把启动时的 grid 从 144 改成 1
//
// （v0 是反面教材，见 reduction_v0_uncoalesced.cu）
#include "kernels/03_reduction/reduction.cuh"
#include "kernels/03_reduction/reduction_block.cuh"

#include "kernels/common/cuda_check.h"

namespace ops_lab {
namespace reduction {
namespace {

template <typename Op>
__global__ void reduce_v1_naive_kernel(const float* __restrict__ x, float* __restrict__ out,
                                       int64_t n) {
  // ① 读：相邻线程读相邻地址 → 合并访存。
  //    每线程约 400 多轮循环，每轮只有一次 load 和一次累加，看起来 ILP 很低；
  //    但循环的不同迭代之间除了 acc 之外没有依赖，硬件可以把多个 load
  //    同时在飞。这一点在第 5 段用 ncu 确认。
  float acc = Op::kIdentity;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    acc = Op::combine(acc, x[i]);
  }

  // ② 合并：块内 smem 顺序寻址树
  const float block_total = block_reduce_smem_tree<Op>(acc);

  // ③ 汇总：每个 block 一次原子操作（本机共 144 次）
  if (threadIdx.x == 0) {
    Op::atomic_combine(out, block_total);
  }
}

}  // namespace

void sum_v1_naive(const float* x, float* out, int64_t n, cudaStream_t stream) {
  init_reduction_output(out, SumOp::kIdentity, stream);
  if (n <= 0) {
    return;
  }
  reduce_v1_naive_kernel<SumOp><<<grid_for_reduction(n), kBlockSize, 0, stream>>>(x, out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

void max_v1_naive(const float* x, float* out, int64_t n, cudaStream_t stream) {
  init_reduction_output(out, MaxOp::kIdentity, stream);
  if (n <= 0) {
    return;
  }
  reduce_v1_naive_kernel<MaxOp><<<grid_for_reduction(n), kBlockSize, 0, stream>>>(x, out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

}  // namespace reduction
}  // namespace ops_lab
