// kernels/05_transpose/transpose_v3_padded.cu
//
// 第 5 章 · 阶梯第 4 级：**只给共享内存的行宽加 1**（32 → 33）。
//
// 与 v2 的 diff 只有一个模板实参：`<kTileCols>` → `<kTileCols + kPad>`。
// 而它消掉的是本章唯一真实存在的 bank conflict：
//
//     地址 tx*32 + (ty+j)  → bank 全等于 (ty+j)      → 32 路冲突
//     地址 tx*33 + (ty+j)  → bank = (tx+ty+j) % 32   → 两两不同 ✓
//
// 预期：**快 5~25%**（推导见 transpose.cuh 的文件头），不是教科书里的 2~4 倍 ——
// 因为共享内存比显存快 24 倍，即使被 32 路冲突打回原形也还剩 1.4 倍余量。
//
// 这一级和第 3 章的 v2 形成对照：那里的 padding 是**真·空操作**（顺序寻址本来
// 就没有冲突），这里的冲突是真的。合起来才能说清"经典优化什么时候有用"。
#include "kernels/05_transpose/transpose.cuh"

#include "kernels/05_transpose/transpose_tiled.cuh"

namespace ops_lab {
namespace transpose {

void transpose_v3_padded(const float* in, float* out, int rows, int cols, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) {
    return;
  }
  const dim3 block(kBlockCols, kBlockRows);
  transpose_tiled_kernel<kTileCols + kPad><<<tiled_grid(rows, cols), block, 0, stream>>>(
      in, out, rows, cols);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

}  // namespace transpose
}  // namespace ops_lab
