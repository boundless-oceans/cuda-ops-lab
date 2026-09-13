// kernels/02_elementwise/elementwise_v4_vectorized.cu
//
// 第 2 章 · 阶梯第 5 级：用 float4 把访存宽度从 4 字节提到 16 字节。
//
// 一次访存搬 16 字节而不是 4 字节，**访存指令数降 4 倍**。在 memory-bound
// 场景下，这一级通常还能再拿 10 个百分点左右。
//
// 刻意**不加** grid-stride：v4 = v1 + 向量化，只改一个变量，
// 这样阶梯表里 v1 → v4 的差距才能干净地归因于访存宽度。
//
// ## 四个边界风险（本章唯一真正容易写错的地方）
//
//   1. n 不是 4 的倍数      → 尾部 0~3 个元素要走标量路径
//   2. 指针不是 16 字节对齐  → **退回标量路径，而不是报错**（见下方 launcher）
//   3. n < 4               → n4 = 0，vector 部分无事可做，但 grid 至少要 1 个
//                            block，否则尾部无人处理
//   4. grid 覆盖不到尾巴     → 尾部复用线程 0~2 的下标（见 kernel 里的 t）
#include "kernels/02_elementwise/elementwise.cuh"

#include "kernels/02_elementwise/elementwise_ops.cuh"
#include "kernels/common/cuda_check.h"

namespace ops_lab {
namespace elementwise {
namespace {

// 向量部分：一个线程负责一个 float4（4 个 float）。
// 尾部：n 不是 4 的倍数时，多出来的 0~3 个元素。
//
// 尾部巧用同一个下标 i：t = n4*4 + i，只有 block 0 的前几个线程会命中
// （因为 n - 4*n4 < 4）。这样不需要第二次 kernel 启动。
//
// 关于签名：**只接收 float4 指针，标量视图在 kernel 内部派生**。
// 不能把 `out4` 和 `out` 作为两个参数一起传进来 —— 它们指向同一块内存，
// 而两者都标了 __restrict__（"我保证没有别名"），同时传就是违反承诺的
// 未定义行为。实测编译器会直接给出警告：
//     warning: passing argument 2 to 'restrict'-qualified parameter aliases
//              with argument 4 [-Wrestrict]
// 从 restrict 指针派生出来的指针不受此限制，所以这样写既干净又安全。
__global__ void add_v4_kernel(const float4* __restrict__ x4, const float4* __restrict__ y4,
                              float4* __restrict__ out4, int64_t n4, int64_t n) {
  const float* x = reinterpret_cast<const float*>(x4);
  const float* y = reinterpret_cast<const float*>(y4);
  float* out = reinterpret_cast<float*>(out4);

  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (i < n4) {
    const float4 a = x4[i];
    const float4 b = y4[i];
    out4[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
  }

  const int64_t t = n4 * 4 + i;
  if (t < n) {
    out[t] = x[t] + y[t];
  }
}

template <typename Op>
__global__ void unary_v4_kernel(const float4* __restrict__ x4, float4* __restrict__ out4,
                                int64_t n4, int64_t n) {
  const float* x = reinterpret_cast<const float*>(x4);
  float* out = reinterpret_cast<float*>(out4);

  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (i < n4) {
    const float4 a = x4[i];
    out4[i] = make_float4(Op::apply(a.x), Op::apply(a.y), Op::apply(a.z), Op::apply(a.w));
  }

  const int64_t t = n4 * 4 + i;
  if (t < n) {
    out[t] = Op::apply(x[t]);
  }
}

// grid 按 float4 的个数算，所以 block 数只有 v1 的 1/4。
// grid_for 保证至少返回 1（n4 == 0 时也要有 block 去处理尾部）。
int64_t grid_for_v4(int64_t n4) {
  return grid_for(n4, kBlockSize, /*allow_grid_stride=*/false);
}

}  // namespace

void add_v4_vectorized(const float* x, const float* y, float* out, int64_t n,
                       cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  // float4 要求 16 字节对齐。不满足就退回 v1 的标量路径。
  //
  // 为什么是"退回"而不是"报错"：非对齐的指针是完全合法的输入。给它报错，
  // 等于把"实现细节的约束"变成"用户的错误" —— 比赛里这种地方最容易丢分。
  if (!is_aligned_16(x) || !is_aligned_16(y) || !is_aligned_16(out)) {
    add_v1_naive(x, y, out, n, stream);
    return;
  }

  const int64_t n4 = n / 4;
  const float4* x4 = reinterpret_cast<const float4*>(x);
  const float4* y4 = reinterpret_cast<const float4*>(y);
  float4* out4 = reinterpret_cast<float4*>(out);

  add_v4_kernel<<<static_cast<unsigned int>(grid_for_v4(n4)), kBlockSize, 0, stream>>>(x4, y4, out4,
                                                                                        n4, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

void relu_v4_vectorized(const float* x, float* out, int64_t n, cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  if (!is_aligned_16(x) || !is_aligned_16(out)) {
    relu_v1_naive(x, out, n, stream);
    return;
  }

  const int64_t n4 = n / 4;
  unary_v4_kernel<ReluOp><<<static_cast<unsigned int>(grid_for_v4(n4)), kBlockSize, 0, stream>>>(
      reinterpret_cast<const float4*>(x), reinterpret_cast<float4*>(out), n4, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

void sigmoid_v4_vectorized(const float* x, float* out, int64_t n, cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  if (!is_aligned_16(x) || !is_aligned_16(out)) {
    sigmoid_v1_naive(x, out, n, stream);
    return;
  }

  const int64_t n4 = n / 4;
  unary_v4_kernel<SigmoidOp><<<static_cast<unsigned int>(grid_for_v4(n4)), kBlockSize, 0,
                               stream>>>(reinterpret_cast<const float4*>(x),
                                         reinterpret_cast<float4*>(out), n4, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

}  // namespace elementwise
}  // namespace ops_lab
