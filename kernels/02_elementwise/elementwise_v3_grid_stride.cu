// kernels/02_elementwise/elementwise_v3_grid_stride.cu
//
// 第 2 章 · 阶梯第 4 级：grid 固定成"刚好填满机器"，kernel 内循环步进。
//
// 与 v1_naive 的关系：v1 的 grid = ceil(n / block)，也就是"有多少元素就开多少
// block"。v3 把 grid 固定成 SM 数 × 每 SM 能驻留的 block 数，每个线程靠循环
// 步进处理多个元素。
//
// ## 确定的收益（功能性）
//
// v1 / v2 的 grid 都受 gridDim.x 的 2^31-1 上限约束，超出就抛错。
// v3 的循环步进天然覆盖任意大的 n，**没有上限**。
//
// ## 不确定的收益（性能）
//
// "消除尾部空转"这个理由对**大规模**输入其实很弱：n = 2^24、block = 256 时
// grid 有 65536 个 block，而机器一次只能驻留 100 多个，要跑几百个 wave。
// 最后一个没填满的 wave 只占总量百分之零点几。
//
// 所以 v3 相对 v1/v2 的性能提升可能很小、甚至没有。**这正是要实测的**
// —— 如果实测三者带宽相当，那说明"grid-stride 更快"这个流传很广的说法
// 在纯流式场景下并不成立，这本身就是有价值的结论。
//
// 反过来，如果 v3 明显更快，那原因多半不是"尾部空转"，而是循环带来的
// 额外 ILP（每次迭代的访存互相独立，编译器可以把它们排开）。
// 判断依据可以看 SASS 里 load 的排布，以及 ncu 的 long scoreboard。
#include "kernels/02_elementwise/elementwise.cuh"

#include "kernels/02_elementwise/elementwise_ops.cuh"
// v3 是本章第一个需要设备信息的变体：它要按 SM 数算 grid 大小。
#include "kernels/common/cuda_check.h"
#include "kernels/common/device.h"

namespace ops_lab {
namespace elementwise {
namespace {

// 全局线程 id 映射一个元素；步长是整个 grid 的线程总数。
// 这就是 grid-stride 循环：每个线程负责 i, i+stride, i+2*stride, ...
//
// 注意这里**不需要额外的尾部处理**：循环条件 i < n 天然覆盖任意 n，
// 包括 n 不是 block 整数倍、n 小于一个 block 的情况。
__global__ void add_v3_kernel(const float* __restrict__ x, const float* __restrict__ y,
                              float* __restrict__ out, int64_t n) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    out[i] = x[i] + y[i];
  }
}

template <typename Op>
__global__ void unary_v3_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t n) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    out[i] = Op::apply(x[i]);
  }
}

// grid 大小由 device.h 的 default_grid_size() 按设备规模算出：
// SM 数 × 每 SM 能同时驻留的 block 数。
//
// 为什么不把 grid 作为参数暴露出来：那会破坏所有变体 launcher 的签名统一
// （run_binary / run_unary 按 5 参数调用），而"统一签名"让绑定层可以对所有
// 变体复用同一段代码 —— 这个价值大于一个暂时用不上的调参旋钮。
// 真要扫 grid 大小做实验时再加。

}  // namespace

void add_v3_grid_stride(const float* x, const float* y, float* out, int64_t n,
                        cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  // 这里**没有** grid 上限检查 —— 因为循环步进让上限不再存在。
  add_v3_kernel<<<default_grid_size(kBlockSize), kBlockSize, 0, stream>>>(x, y, out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

void relu_v3_grid_stride(const float* x, float* out, int64_t n, cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  unary_v3_kernel<ReluOp><<<default_grid_size(kBlockSize), kBlockSize, 0, stream>>>(x, out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

void sigmoid_v3_grid_stride(const float* x, float* out, int64_t n, cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  unary_v3_kernel<SigmoidOp><<<default_grid_size(kBlockSize), kBlockSize, 0, stream>>>(x, out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

}  // namespace elementwise
}  // namespace ops_lab
