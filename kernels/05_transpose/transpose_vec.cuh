// kernels/05_transpose/transpose_vec.cuh
//
// v4 / v5 共用的向量化转置 kernel。**模板参数只有瓦片宽度**：
//
//     v4_vectorized → transpose_vec_kernel<kTileCols, kTileCols + kVecPad>       (32)
//     v5_wide_tile  → transpose_vec_kernel<kWideTileCols, kWideTileCols + kVecPad> (64)
//
// ## 线程映射（为什么向量化一定会改映射）
//
//     block = (8, 32) = 256 线程
//     x 方向 8 个线程 × 4 个连续列 = 瓦片宽度的 1/2 或 1/4，用 kColPasses 补齐
//     y 方向 32 个线程 = 32 行
//
// 每个线程一次搬一个 **float4**：
//     载入：读 4 个连续**列**（同一行）—— 全局合并 ✓
//     写回：写 4 个连续**输出列** = 4 个连续**输入行** —— 全局合并 ✓
//           这 4 个值在 smem 里是 4 个跨步行（步长 = 行宽），所以是 4 次标量读。
//
// **共享内存仍然用标量读写**，所以行宽 +1 就足以消 bank conflict：
//     读的地址 = (4·tx + k) · (W+1) + (p·32 + ty)，W+1 与 32 互质
//     → bank = (4·tx + k + p·32 + ty) mod 32，一个 warp 内两两不同 ✓
// 如果共享内存也要 float4 读写，行宽就必须是 4 的倍数，而 4 的倍数与 32 不互质
// → padding 失效，得换 XOR swizzle（见 transpose_config.cuh 的注释）。
//
// ## 边界
//
// 快路径要求**两件事**：① 4 个连续元素都在界内；② 指针 16 字节对齐。
//
// ② 特别容易漏：`in_col` 是 4 的倍数，但**行起点 `in_row * cols` 只有在
// `cols % 4 == 0` 时才对齐** —— 列数不是 4 的倍数时，从第二行起 float4 就错位。
// 实测漏掉它的后果是 `CUDA error: misaligned address`，而且**脏上下文会让
// 之后所有测试都报同一个错**（连没有 float4 的 v0 也'失败'），非常误导。
// 判据由 host 侧算好传进来（见两个 launcher 里的 vec_ok）。
//
// 快路径的两级边界条件还是互相对应的
// （载入时的 `in_col+k < cols` ⟺ 写回时的 `out_row < cols`），所以在部分瓦片上
// 不会读到没写过的格子。
#pragma once

#include "kernels/05_transpose/transpose_config.cuh"

namespace ops_lab {
namespace transpose {

template <int kTileW, int kStride>
__global__ void transpose_vec_kernel(const float* __restrict__ in, float* __restrict__ out,
                                     int rows, int cols, int vec_ok) {
  __shared__ float tile[kTileRows][kStride];

  constexpr int kPassCols = kVecBlockCols * kVecItems;   // 一趟覆盖多少列（32）
  constexpr int kColPasses = kTileW / kPassCols;         // 列方向要几趟（32→1，64→2）

  const int in_row = blockIdx.y * kTileRows + threadIdx.y;

  // ① 载入：每趟一个 float4（4 个连续列）
  if (in_row < rows) {
#pragma unroll
    for (int p = 0; p < kColPasses; ++p) {
      const int in_col = blockIdx.x * kTileW + p * kPassCols + threadIdx.x * kVecItems;
      const int tile_c = p * kPassCols + threadIdx.x * kVecItems;
      if (vec_ok && in_col + kVecItems <= cols) {
        const float4 v = *reinterpret_cast<const float4*>(in + in_row * cols + in_col);
        tile[threadIdx.y][tile_c + 0] = v.x;
        tile[threadIdx.y][tile_c + 1] = v.y;
        tile[threadIdx.y][tile_c + 2] = v.z;
        tile[threadIdx.y][tile_c + 3] = v.w;
      } else {
        for (int k = 0; k < kVecItems; ++k) {
          if (in_col + k < cols) {
            tile[threadIdx.y][tile_c + k] = in[in_row * cols + in_col + k];
          }
        }
      }
    }
  }
  __syncthreads();

  // ③ 写回：输出行 = 输入列（kTileW 个 → kColPasses 趟），输出列 = 输入行
  const int out_col0 = blockIdx.y * kTileRows + threadIdx.x * kVecItems;
#pragma unroll
  for (int p = 0; p < kColPasses; ++p) {
    const int out_row = blockIdx.x * kTileW + p * kTileRows + threadIdx.y;
    const int tile_r = p * kTileRows + threadIdx.y;
    if (out_row < cols) {
      if (vec_ok && out_col0 + kVecItems <= rows) {
        float4 r;
        r.x = tile[threadIdx.x * kVecItems + 0][tile_r];
        r.y = tile[threadIdx.x * kVecItems + 1][tile_r];
        r.z = tile[threadIdx.x * kVecItems + 2][tile_r];
        r.w = tile[threadIdx.x * kVecItems + 3][tile_r];
        *reinterpret_cast<float4*>(out + out_row * rows + out_col0) = r;
      } else {
        for (int k = 0; k < kVecItems; ++k) {
          if (out_col0 + k < rows) {
            out[out_row * rows + out_col0 + k] = tile[threadIdx.x * kVecItems + k][tile_r];
          }
        }
      }
    }
  }
}

}  // namespace transpose
}  // namespace ops_lab
