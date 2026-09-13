// kernels/02_elementwise/elementwise_v2_unroll4.cu
//
// 第 2 章 · 阶梯第 3 级：每线程处理 4 个元素，靠指令级并行（ILP）提高
// 每线程的**在飞访存请求数**。
//
// v1_naive 的问题不是访存没合并（它已经合并了），而是**并发度不够**：
// 每个线程只有 1 个访存在飞，访存延迟只能靠"线程数多"来掩盖。
// 而 SM 能容纳的线程数是有限的（受寄存器和 block 数限制）。
//
// v2 换一条路：线程数降到 1/4，但每个线程一口气发出 4 个**互不依赖**的
// 访存请求。总并发度 = 线程数 × 每线程在飞请求数，这一项补回来了。
//
// 为什么是"两段式"（先全部 load，再全部 store）：
//   如果把 `if (i < n)` 塞进循环里，编译器无法确定 4 个 load 是否都会执行，
//   只能一个一个来 —— ILP 就没了。所以拆成快慢两条路径：绝大多数 block 走
//   快路径（4 个下标都合法，可以放心地把 load 提前），只有最后一个 block
//   走慢路径逐元素检查。
//
// 代价：寄存器占用上升（要同时装 4 个 x 和 4 个 y），这是 ILP 的固有代价。
#include "kernels/02_elementwise/elementwise.cuh"

#include "kernels/02_elementwise/elementwise_ops.cuh"
#include "kernels/common/cuda_check.h"

namespace ops_lab {
namespace elementwise {
namespace {

// 每线程处理的元素个数。调大能提高 ILP，但寄存器压力线性上升，
// 到某个点会因为 occupancy 下降而反噬 —— 这正是本章值得实验的地方。
constexpr int kUnroll = 4;

// 一个 block 负责连续的 blockDim.x * kUnroll 个元素；
// 线程 t 负责其中的第 t、t+B、t+2B、t+3B 个（B = blockDim.x）。
// 这 4 个位置在各自 warp 内**都是相邻线程访问相邻地址** → 依然是合并访存。
__device__ __forceinline__ int64_t unroll_base() {
  return static_cast<int64_t>(blockIdx.x) * (blockDim.x * kUnroll) + threadIdx.x;
}

// 4 个下标是否全部落在 [0, n)
__device__ __forceinline__ bool full_chunk(int64_t base, int64_t n) {
  return base + static_cast<int64_t>(blockDim.x) * (kUnroll - 1) < n;
}

__global__ void add_v2_kernel(const float* __restrict__ x, const float* __restrict__ y,
                              float* __restrict__ out, int64_t n) {
  const int64_t base = unroll_base();
  const int64_t stride = blockDim.x;

  if (full_chunk(base, n)) {
    // 快路径：先把 4 个 x 和 4 个 y 全部读进来（8 个独立 load），再统一写回。
    // 两个循环分开写，编译器才能把这 8 个 load 排在一起发射。
    float a[kUnroll];
    float b[kUnroll];
#pragma unroll
    for (int k = 0; k < kUnroll; ++k) {
      a[k] = x[base + k * stride];
    }
#pragma unroll
    for (int k = 0; k < kUnroll; ++k) {
      b[k] = y[base + k * stride];
    }
#pragma unroll
    for (int k = 0; k < kUnroll; ++k) {
      out[base + k * stride] = a[k] + b[k];
    }
    return;
  }

  // 慢路径：只有最后一个 block 会走到这里。
#pragma unroll
  for (int k = 0; k < kUnroll; ++k) {
    const int64_t i = base + k * stride;
    if (i < n) {
      out[i] = x[i] + y[i];
    }
  }
}

template <typename Op>
__global__ void unary_v2_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t n) {
  const int64_t base = unroll_base();
  const int64_t stride = blockDim.x;

  if (full_chunk(base, n)) {
    float a[kUnroll];
#pragma unroll
    for (int k = 0; k < kUnroll; ++k) {
      a[k] = x[base + k * stride];
    }
#pragma unroll
    for (int k = 0; k < kUnroll; ++k) {
      out[base + k * stride] = Op::apply(a[k]);
    }
    return;
  }

#pragma unroll
  for (int k = 0; k < kUnroll; ++k) {
    const int64_t i = base + k * stride;
    if (i < n) {
      out[i] = Op::apply(x[i]);
    }
  }
}

// 每个 block 覆盖 kUnroll 倍的元素，所以 block 数只有 v1 的 1/4。
// 与 v1 一样保留 grid 上限检查 —— 解除它是 v3_grid_stride 的事。
int64_t grid_for_v2(int64_t n) {
  return grid_for(n, static_cast<int64_t>(kBlockSize) * kUnroll,
                  /*allow_grid_stride=*/false);
}

}  // namespace

void add_v2_unroll4(const float* x, const float* y, float* out, int64_t n, cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  add_v2_kernel<<<static_cast<unsigned int>(grid_for_v2(n)), kBlockSize, 0, stream>>>(x, y, out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

void relu_v2_unroll4(const float* x, float* out, int64_t n, cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  unary_v2_kernel<ReluOp><<<static_cast<unsigned int>(grid_for_v2(n)), kBlockSize, 0, stream>>>(
      x, out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

void sigmoid_v2_unroll4(const float* x, float* out, int64_t n, cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  unary_v2_kernel<SigmoidOp><<<static_cast<unsigned int>(grid_for_v2(n)), kBlockSize, 0,
                               stream>>>(x, out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

}  // namespace elementwise
}  // namespace ops_lab
