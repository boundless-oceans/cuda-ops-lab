// kernels/05_transpose/transpose_v5_wide_tile.cu
//
// 第 5 章 · 阶梯第 6 级：与 v4 只差**瓦片宽度**（32 → 64）。
//
//      `transpose_vec_kernel<32, 33>`  →  `transpose_vec_kernel<64, 65>`
//
// 为什么这可能真的有用：每个 block 一次连续访问的字节数从 **128 B** 变成 **256 B**。
// 128 B 恰好是一个 sector 段的边界，而 256 B 跨两个段 —— DRAM 的行缓冲命中率会跟着变。
// 这和第 1 章 `probe_smem48k` 那个未解释现象同源：**并发流少一点、连续段长一点，
// 显存效率就高一点。**
//
// block 大小和线程映射都没变（(8,32)），只有瓦片宽了 —— 于是每个线程的列方向
// 从 1 趟变 2 趟（kColPasses 由模板算出来）。**这是一个干净的"只改宽度"。**
//
// 预期：比 v4 好 5~15%。**不确定，而且很可能测不出来** —— 那就如实记下来，
// 并把"瓦片宽度 vs DRAM 突发"这条假设标成未验证。
#include "kernels/05_transpose/transpose.cuh"

#include "kernels/05_transpose/transpose_vec.cuh"

namespace ops_lab {
namespace transpose {

void transpose_v5_wide_tile(const float* in, float* out, int rows, int cols,
                            cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) {
    return;
  }
  const dim3 block(kVecBlockCols, kVecBlockRows);
  const dim3 grid(static_cast<unsigned int>(ceil_div(cols, kWideTileCols)),
                  static_cast<unsigned int>(ceil_div(rows, kTileRows)));
  // float4 的两个前提：4 个连续元素在界内（kernel 里判），以及 **16 字节对齐**。
  // 行起点是 `in_row * cols` / `out_row * rows`，所以列数和行数都必须是 4 的倍数 ——
  // 漏掉这一条会得到 `misaligned address`，而且会污染上下文让别的变体也报错。
  const int vec_ok =
      (is_aligned_16(in) && is_aligned_16(out) && (cols % 4 == 0) && (rows % 4 == 0)) ? 1 : 0;
  transpose_vec_kernel<kWideTileCols, kWideTileCols + kVecPad><<<grid, block, 0, stream>>>(
      in, out, rows, cols, vec_ok);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

}  // namespace transpose
}  // namespace ops_lab
