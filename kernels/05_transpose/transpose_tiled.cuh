// kernels/05_transpose/transpose_tiled.cuh
//
// v2 / v3 共用的 kernel：**模板参数只有共享内存的行宽**。
//
//     v2_smem_tiled → transpose_tiled_kernel<kTileCols>            (32)
//     v3_padded     → transpose_tiled_kernel<kTileCols + kPad>     (33)
//
// 两个变体的 diff 因此只有一个模板实参 —— 而它正好就是本章唯一要证的那件事。
//
// ## 内核结构（经典形态）
//
//     block = (32, 8) = 256 线程；瓦片 32×32；每个线程负责 4 行（跨步 8 行）
//
//     ① tile[ty+j][tx] = in[(y+j)*cols + x]      ← 相邻 tx 相邻地址：读合并
//                                                  写 smem 行内：无冲突
//     ② __syncthreads()
//     ③ out[(y2+j)*rows + x2] = tile[tx][ty+j]   ← 写全局合并
//                                                  **读 smem 列：32 路冲突**（v2）
//
// 第 ③ 步的两组下标是"交叉"的：输出行 = 输入列，输出列 = 输入行。
// 写回时的两个边界条件也是交叉的（`out_row+j < cols && out_col < rows`），
// 与载入时的 `in_row+j < rows && in_col < cols` 恰好互为转置 —— 所以在部分瓦片
// （M 或 N 不是 32 的倍数）上，**没被写入的 smem 格子也不会被读出来**。
#pragma once

#include "kernels/05_transpose/transpose_config.cuh"

namespace ops_lab {
namespace transpose {

// kStride = 共享内存的一行有多少个 float（v2 是 32，v3 是 33）
template <int kStride>
__global__ void transpose_tiled_kernel(const float* __restrict__ in, float* __restrict__ out,
                                       int rows, int cols) {
  __shared__ float tile[kTileRows][kStride];

  const int in_col = blockIdx.x * kTileCols + threadIdx.x;
  const int in_row = blockIdx.y * kTileRows + threadIdx.y;

  // ① 载入：float 读全局是合并的（相邻 tx → 相邻列）
#pragma unroll
  for (int j = 0; j < kTileRows; j += kBlockRows) {
    if (in_row + j < rows && in_col < cols) {
      tile[threadIdx.y + j][threadIdx.x] = in[(in_row + j) * cols + in_col];
    }
  }
  __syncthreads();

  // ③ 写回：输出行 = 输入列，输出列 = 输入行
  const int out_col = blockIdx.y * kTileRows + threadIdx.x;
  const int out_row = blockIdx.x * kTileCols + threadIdx.y;
#pragma unroll
  for (int j = 0; j < kTileRows; j += kBlockRows) {
    if (out_row + j < cols && out_col < rows) {
      // ★ 这一行是本章的主角：kStride = 32 时，一个 warp 的 32 个线程算出的地址
      //   是 tx*32 + (ty+j) —— 模 32 全等于 (ty+j)，**32 路冲突**。
      //   kStride = 33 时变成 tx*33 + (ty+j) → 模 32 等于 (tx+ty+j)，两两不同。
      out[(out_row + j) * rows + out_col] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

}  // namespace transpose
}  // namespace ops_lab
