// kernels/05_transpose/transpose_v1_swapped.cu
//
// 第 5 章 · 阶梯第 2 级：反面教材之二 —— **读跨步**、写合并。
//
// 与 v0 **只差交换 (r, c) 的取法**（以及跟着变的 grid 形状）：
// 让相邻线程沿 r 走，于是 out[c*rows + r] 相邻地址、写合并；而 in[r*cols + c]
// 变成跨步读。
//
// 预期：**比 v0 更差**。这个不对称是有道理的：
//   * 读的数据要立刻拿到才能算，延迟藏不住 → 跨步读的 32 个事务全部要等
//   * 写只要进了回写缓冲就算完事，L2 还有机会把它合并
// 把这一级量出来，"读写不对称"就不再是一句口号。
#include "kernels/05_transpose/transpose.cuh"

namespace ops_lab {
namespace transpose {
namespace {

__global__ void transpose_v1_kernel(const float* __restrict__ in, float* __restrict__ out,
                                    int rows, int cols) {
  // ★ 与 v0 只差这两行：把 blockIdx/threadIdx 的 y 给 c、x 给 r
  const int c = blockIdx.y * kNaiveBlock + threadIdx.y;
  const int r = blockIdx.x * kNaiveBlock + threadIdx.x;
  if (r < rows && c < cols) {
    out[c * rows + r] = in[r * cols + c];
  }
}

}  // namespace

void transpose_v1_swapped(const float* in, float* out, int rows, int cols, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) {
    return;
  }
  const dim3 block(kNaiveBlock, kNaiveBlock);
  // grid 也跟着调换：x 方向现在铺的是行
  const dim3 grid(static_cast<unsigned int>(ceil_div(rows, kNaiveBlock)),
                  static_cast<unsigned int>(ceil_div(cols, kNaiveBlock)));
  transpose_v1_kernel<<<grid, block, 0, stream>>>(in, out, rows, cols);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

}  // namespace transpose
}  // namespace ops_lab
