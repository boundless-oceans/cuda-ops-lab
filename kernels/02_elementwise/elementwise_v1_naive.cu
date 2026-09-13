// kernels/02_elementwise/elementwise_v1_naive.cu
//
// 第 2 章 · 阶梯第 2 级：一线程一元素、连续访问（已合并访存）。
//
// 这是整条阶梯的**基准点** —— 它已经是"正确且合并访存"的写法，
// 后面每一级都从这里出发去修一个具体的瓶颈：
//
//   v1 → v2  每线程只有 1 个访存在飞，并发度不足
//   v1 → v3  grid 有上限，且尾部有空转
//   v1 → v4  一次只搬 4 字节，访存指令数太多
//
// （v0 是反面教材，见 elementwise_v0_uncoalesced.cu）
#include "kernels/02_elementwise/elementwise.cuh"

#include "kernels/02_elementwise/elementwise_ops.cuh"
#include "kernels/common/cuda_check.h"

namespace ops_lab {
namespace elementwise {
namespace {

// 全局线程 id 直接映射一个元素：相邻线程访问相邻地址 → 天然合并访存。
// 注意这里没有 grid-stride 循环：i 一次性算出，越界就返回。
// 这正是 v1 与 v3 的核心差别。
__global__ void add_v1_naive_kernel(const float* __restrict__ x,
                                    const float* __restrict__ y,
                                    float* __restrict__ out,
                                    int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = x[i] + y[i];
  }
}

// 一目算子（relu / sigmoid）共用这一个 kernel 模板：
// 访存模式完全相同，只有 Op::apply 里的算术不同。
template <typename Op>
__global__ void unary_v1_naive_kernel(const float* __restrict__ x,
                                      float* __restrict__ out,
                                      int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = Op::apply(x[i]);
  }
}

// allow_grid_stride = false：超出 grid 上限直接抛清晰错误（见 cuda_check.h）。
// 这是 v1 的**固有约束**，不是 bug，所以在启动前就拦下来。解除它是 v3 的事。
int64_t grid_for_v1(int64_t n) { return grid_for(n, kBlockSize, /*allow_grid_stride=*/false); }

}  // namespace

void add_v1_naive(const float* x, const float* y, float* out, int64_t n, cudaStream_t stream) {
  if (n <= 0) {
    return;  // 空输入不启动 kernel
  }
  add_v1_naive_kernel<<<static_cast<unsigned int>(grid_for_v1(n)), kBlockSize, 0, stream>>>(x, y,
                                                                                            out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

void relu_v1_naive(const float* x, float* out, int64_t n, cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  unary_v1_naive_kernel<ReluOp><<<static_cast<unsigned int>(grid_for_v1(n)), kBlockSize, 0,
                                  stream>>>(x, out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

void sigmoid_v1_naive(const float* x, float* out, int64_t n, cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  unary_v1_naive_kernel<SigmoidOp><<<static_cast<unsigned int>(grid_for_v1(n)), kBlockSize, 0,
                                     stream>>>(x, out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

}  // namespace elementwise
}  // namespace ops_lab
