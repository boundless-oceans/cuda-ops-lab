// kernels/02_elementwise/elementwise_v0_uncoalesced.cu
//
// 第 2 章 · 阶梯第 1 级：**反面教材**。
//
// 与 v1_naive 相比，**只有索引公式这一行不同**：
//
//   v1_naive :  i = blockIdx.x * blockDim.x + threadIdx.x    ← 相邻线程相邻地址
//   v0       :  i = threadIdx.x * gridDim.x + blockIdx.x     ← 相邻线程相隔 gridDim.x
//
// 后果：一个 warp 里 32 个线程的地址两两相隔 gridDim.x 个元素，
// 也就是落在 32 条**不同的 cache line** 上。一次访存请求被拆成 32 个事务
// → 有效带宽掉到峰值的 10~20%。
//
// 为什么非要写一个错的版本：因为"未合并访存只有 15% 带宽"这个**实测数字**，
// 比"应该合并访存"这句话有用 10 倍。
//
// 注意这里**没有循环**：grid = ceil(n / block)，而
// i = threadIdx.x * gridDim.x + blockIdx.x 恰好能遍历 [0, blockDim.x * gridDim.x)，
// 覆盖了全部 n 个元素。所以除了索引公式，它和 v1 一模一样 ——
// 这样阶梯表里 v0 → v1 的差距就**只能**归因于访存模式。
#include "kernels/02_elementwise/elementwise.cuh"

#include "kernels/02_elementwise/elementwise_ops.cuh"
#include "kernels/common/cuda_check.h"

namespace ops_lab {
namespace elementwise {
namespace {

// 反面教材的访存模式。抽成函数是为了让 add / relu / sigmoid 三个 kernel
// 用的是同一个"错误"，避免手抄时抄歪了。
__device__ __forceinline__ int64_t uncoalesced_index() {
  return static_cast<int64_t>(threadIdx.x) * gridDim.x + blockIdx.x;
}

__global__ void add_v0_kernel(const float* __restrict__ x, const float* __restrict__ y,
                              float* __restrict__ out, int64_t n) {
  const int64_t i = uncoalesced_index();
  if (i < n) {
    out[i] = x[i] + y[i];
  }
}

template <typename Op>
__global__ void unary_v0_kernel(const float* __restrict__ x, float* __restrict__ out,
                                int64_t n) {
  const int64_t i = uncoalesced_index();
  if (i < n) {
    out[i] = Op::apply(x[i]);
  }
}

// v0 与 v1 一样带 grid 上限检查：这一级的阶梯不负责解决"grid 装不下"的问题，
// 那是 v3_grid_stride 的事。
int64_t grid_for_v0(int64_t n) { return grid_for(n, kBlockSize, /*allow_grid_stride=*/false); }

}  // namespace

void add_v0_uncoalesced(const float* x, const float* y, float* out, int64_t n,
                        cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  add_v0_kernel<<<static_cast<unsigned int>(grid_for_v0(n)), kBlockSize, 0, stream>>>(x, y, out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

void relu_v0_uncoalesced(const float* x, float* out, int64_t n, cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  unary_v0_kernel<ReluOp><<<static_cast<unsigned int>(grid_for_v0(n)), kBlockSize, 0, stream>>>(
      x, out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

void sigmoid_v0_uncoalesced(const float* x, float* out, int64_t n, cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  unary_v0_kernel<SigmoidOp><<<static_cast<unsigned int>(grid_for_v0(n)), kBlockSize, 0, stream>>>(
      x, out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

}  // namespace elementwise
}  // namespace ops_lab
