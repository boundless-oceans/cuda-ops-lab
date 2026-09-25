// kernels/05_transpose/transpose_v2_smem_tiled.cu
//
// 第 5 章 · 阶梯第 3 级：引入 32×32 的共享内存瓦片（**不加 padding**）。
//
// 与 v1 相比换掉了整个结构：读和写都经过 smem 中转，于是**两边都合并**了。
// 代价是写回阶段要读 smem 的**一列**，而一行的宽度恰好是 32 —— 一个 warp 的
// 32 个线程会全部落在同一个 bank 上：**32 路 bank conflict**。
//
// 预期：相对 v1 是**一大步**；但还留着这个冲突没解决（留给 v3）。
#include "kernels/05_transpose/transpose.cuh"

#include "kernels/05_transpose/transpose_tiled.cuh"

namespace ops_lab {
namespace transpose {

void transpose_v2_smem_tiled(const float* in, float* out, int rows, int cols,
                             cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) {
    return;
  }
  const dim3 block(kBlockCols, kBlockRows);
  // kStride = 32：一行恰好铺满 32 个 bank
  transpose_tiled_kernel<kTileCols><<<tiled_grid(rows, cols), block, 0, stream>>>(in, out, rows,
                                                                                 cols);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

}  // namespace transpose
}  // namespace ops_lab
