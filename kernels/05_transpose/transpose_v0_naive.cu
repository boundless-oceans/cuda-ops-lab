// kernels/05_transpose/transpose_v0_naive.cu
//
// 第 5 章 · 阶梯第 1 级：反面教材之一 —— 读合并、**写跨步**。
//
// 线程 (tx, ty) 直接处理 in[r][c]：
//     相邻 tx → 相邻 c → 读天然合并 ✓
//     而 out[c*rows + r] 在相邻 tx 之间隔了 `rows` 个元素 → 一个 warp 落在 32 条
//     cache line 上 ✗
//
// 预期：明显慢，**但不是教科书的 32 倍**。第 2 章的教训在这里同样适用 ——
// 写是"回写"的，L2 能把同一个 sector 的多次写入合并起来，
// 所以代价体现为**事务数**而不是显存带宽。
//
// 它是 v1 的对照组：两者只差"相邻线程沿哪个下标走"。
#include "kernels/05_transpose/transpose.cuh"

namespace ops_lab {
namespace transpose {
namespace {

__global__ void transpose_v0_kernel(const float* __restrict__ in, float* __restrict__ out,
                                    int rows, int cols) {
  const int r = blockIdx.y * kNaiveBlock + threadIdx.y;
  const int c = blockIdx.x * kNaiveBlock + threadIdx.x;
  if (r < rows && c < cols) {
    out[c * rows + r] = in[r * cols + c];
  }
}

}  // namespace

void transpose_v0_naive(const float* in, float* out, int rows, int cols, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) {
    return;
  }
  const dim3 block(kNaiveBlock, kNaiveBlock);
  transpose_v0_kernel<<<naive_grid(rows, cols), block, 0, stream>>>(in, out, rows, cols);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

}  // namespace transpose
}  // namespace ops_lab
