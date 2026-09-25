// kernels/04_scan/scan_kernels.cuh
//
// 三段式（v0~v2 共用）的 kernel 声明，以及三个变体共用的"第三趟" kernel 模板。
//
// ## 为什么第三趟写成模板
//
// 三个变体（朴素 / Kogge-Stone / Blelloch）的第三趟**只差块内扫描那一次调用**，
// 其余（载入、加 chunk 前缀、写回）完全相同。写成模板之后，变体之间的差别就是
// launcher 里的一行模板参数 —— 少抄三遍就少三处不同步。
//
// 反过来说：如果哪天有人给某个变体加了特例，模板会逼他显式写出来，
// 而不是藏在复制粘贴的某一行里。
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "kernels/04_scan/scan_block.cuh"

namespace ops_lab {
namespace scan {

// ---------------------------------------------------------------- 共享的两趟
//
// 这两个 kernel 被 v0~v2 共用（定义在 scan_block_sums.cu）。
// 它们不是模板，所以在头里只声明、不定义 —— 跨 TU 调用 __global__ 函数是允许的，
// 前提是非 RDC 模式下（本仓库的构建就是这么配的）。

// A 趟：每个 block 把一个 chunk 缩成一个和 → sums[blockIdx.x]
__global__ void chunk_sums_kernel(const float* __restrict__ x, float* __restrict__ sums, int64_t n);

// B 趟：**一个 block** 把 chunk 和数组就地扫成 exclusive 前缀
__global__ void scan_chunk_sums_kernel(float* __restrict__ sums, int64_t count);

// ------------------------------------------------------------------ C 趟
//
// 入口 block_prefix[blockIdx.x] = 前面所有 chunk 的和（exclusive），
// 出口 out[gi] = 本 chunk 的 inclusive 前缀 + 前面的 chunk 前缀。
template <typename Policy>
__global__ void scan_apply_kernel(const float* __restrict__ x,
                                  const float* __restrict__ block_prefix,
                                  float* __restrict__ out, int64_t n) {
  __shared__ float smem[kBlockSize + 1];

  const int64_t base = static_cast<int64_t>(blockIdx.x) * kChunkSize;
  const int64_t gi = base + threadIdx.x;
  // 尾部不足一个 chunk 的用 0 补齐 —— scan 的单位元就是 0，
  // 补 0 不会影响前面真实元素的前缀和（这一点和 reduction 完全一样）。
  const float v = (gi < n) ? x[gi] : 0.0f;
  smem[threadIdx.x] = v;

  Policy::scan(smem, v);  // ← 三个变体之间唯一不同的一行

  const float prefix = block_prefix[blockIdx.x];
  if (gi < n) {
    out[gi] = smem[threadIdx.x] + v + prefix;
  }
}

}  // namespace scan
}  // namespace ops_lab
