// kernels/04_scan/scan_three_pass.cuh
//
// 三段式（scan-scan-add）的**公共流程**。
//
// v0 / v1 / v2 三个变体的唯一差别，就是这个函数的模板参数 `Policy`
// （也就是"第三趟用哪种块内扫描"）。把流程放在这里，变体文件里就只剩
// 一句实例化 —— 邻接 diff 就是那一句。
//
// 代价要说清楚：这个流程**读了两遍 x**（A 趟和 C 趟各一遍），
// 所以它的理论下限是 12N 而不是 8N。把这一遍省掉是 v3 存在的全部理由。
#pragma once

#include <stdexcept>
#include <string>

#include "kernels/04_scan/scan.cuh"
#include "kernels/04_scan/scan_kernels.cuh"
#include "kernels/common/cuda_check.h"

namespace ops_lab {
namespace scan {

// A 趟：读 x → 每个 chunk 缩成一个和（grid = chunk 数）
// B 趟：**一个 block** 把 chunk 和就地扫成 exclusive 前缀（数据量 N/256，可忽略）
// C 趟：再读一遍 x → 块内扫描 + 加 chunk 前缀 → 写 out（grid = chunk 数）
//
// workspace：前 `chunk_count(n)` 个 float 装 chunk 前缀。三段式不需要
// aggregate/inclusive/status 那三段 —— 那是单趟实现（v3）才要的。
template <typename Policy>
inline void run_three_pass(const float* x, float* out, int64_t n, float* workspace,
                           cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  const int64_t chunks = chunk_count(n);
  if (workspace == nullptr) {
    throw std::runtime_error(
        "scan 三段式需要 workspace（每 chunk 一个 float 的前缀数组），"
        "传 nullptr 是调用方的错误：\n  n = " + std::to_string(n) + " → 需要 " +
        std::to_string(chunks) + " 个 float。绑定层与 native 线都要自己分配。");
  }
  const unsigned int grid = grid_for_chunks(n);

  chunk_sums_kernel<<<grid, kBlockSize, 0, stream>>>(x, workspace, n);
  OPSLAB_CUDA_CHECK_LAUNCH();

  scan_chunk_sums_kernel<<<1, kBlockSize, 0, stream>>>(workspace, chunks);
  OPSLAB_CUDA_CHECK_LAUNCH();

  scan_apply_kernel<Policy><<<grid, kBlockSize, 0, stream>>>(x, workspace, out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

}  // namespace scan
}  // namespace ops_lab
