// kernels/05_transpose/transpose_v4_vectorized.cu
//
// 第 5 章 · 阶梯第 5 级：与 v3 只差**访存宽度** —— 读写两侧都用 float4。
//
//      `transpose_tiled_kernel<33>`  →  `transpose_vec_kernel<32, 33>`
//
// 注意线程映射**必须**跟着改（block 从 (32,8) 变成 (8,32)）：float4 要求每个线程
// 覆盖 4 个连续元素，这是向量化的固有代价，不是疏忽 —— 第 2 章的 v4 也有同样的
// 映射变化，那一章它反而**更慢**（省下的是指令，不是时间）。
//
// 预期：与 v3 打平或略差。若实测明显更快，说明 v3 还没到显存上限。
#include "kernels/05_transpose/transpose.cuh"

#include "kernels/05_transpose/transpose_vec.cuh"

namespace ops_lab {
namespace transpose {

void transpose_v4_vectorized(const float* in, float* out, int rows, int cols,
                             cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) {
    return;
  }
  const dim3 block(kVecBlockCols, kVecBlockRows);
  const dim3 grid(static_cast<unsigned int>(ceil_div(cols, kTileCols)),
                  static_cast<unsigned int>(ceil_div(rows, kTileRows)));
  // float4 的两个前提：4 个连续元素在界内（kernel 里判），以及 **16 字节对齐**。
  // 行起点是 `in_row * cols` / `out_row * rows`，所以列数和行数都必须是 4 的倍数 ——
  // 漏掉这一条会得到 `misaligned address`，而且会污染上下文让别的变体也报错。
  const int vec_ok =
      (is_aligned_16(in) && is_aligned_16(out) && (cols % 4 == 0) && (rows % 4 == 0)) ? 1 : 0;
  transpose_vec_kernel<kTileCols, kTileCols + kVecPad><<<grid, block, 0, stream>>>(in, out, rows, cols, vec_ok);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

}  // namespace transpose
}  // namespace ops_lab
