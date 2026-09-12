// kernels/02_elementwise/elementwise_v1_naive.cu
//
// 第 2 章 · 阶梯第 2 级：一线程一元素、连续访问（已合并访存）。
#include "kernels/02_elementwise/elementwise.cuh"

#include "kernels/common/cuda_check.h"

namespace ops_lab {
namespace elementwise {
namespace {

// 注意这里没有 grid-stride 循环：i 一次性算出，越界就返回。
// 这是 v1 与 v3 的核心差别（v3 会循环步进，从而摆脱 grid 上限）。
__global__ void add_v1_naive_kernel(const float* __restrict__ x,
                                    const float* __restrict__ y,
                                    float* __restrict__ out,
                                    int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = x[i] + y[i];
  }
}

}  // namespace

void add_v1_naive(const float* x, const float* y, float* out, int64_t n, cudaStream_t stream) {
  if (n <= 0) {
    return;  // 空输入不启动 kernel（n == 0 时启动也是合法的空 kernel，但没必要）
  }

  // allow_grid_stride = false：超出 grid 上限直接抛清晰错误（见 cuda_check.h）。
  // 这是 v1 的**固有约束**，不是 bug，所以在启动前就拦下来。
  const int64_t grid = grid_for(n, kBlockSize, /*allow_grid_stride=*/false);

  add_v1_naive_kernel<<<static_cast<unsigned int>(grid), kBlockSize, 0, stream>>>(x, y, out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

}  // namespace elementwise
}  // namespace ops_lab
