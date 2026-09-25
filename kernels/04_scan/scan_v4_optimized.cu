// kernels/04_scan/scan_v4_optimized.cu
//
// 第 4 章 · 阶梯第 5 级：把两处修法合起来 —— 单趟 look-back + **每线程 K = 8 个元素**。
//
// 与 v3 相比只改了一件事：**chunk 从 256 放大到 2048**（每线程 8 个元素）。
// 代码上就是模板参数从 1 变成 8，其余（结构、回看、块内算法）完全相同。
//
// 为什么要摊销（第一版实测逼出来的）：一个 chunk 只需要 8 次 barrier，但 v3 拿它
// 服务 256 个元素（每元素 0.031 次 barrier），v4 服务 2048 个（0.0039 次）——除以 8。
// 第一版 v1 的 12N 流量只值 427 µs 却实测 727.8 µs，差的 300 µs 里绝大部分就是这项。
// 详见 scan_chunked.cuh 的文件头。
//
// 预期：**~85~90% 峰值（按 8N 算），约 300~330 µs** —— 也就是 torch 现在的位置
// （实测 294.9 µs / 88.9%）。能不能打平是本级唯一的判据。
#include "kernels/04_scan/scan.cuh"

#include "kernels/04_scan/scan_single_pass.cuh"

namespace ops_lab {
namespace scan {

void scan_inclusive_v4_optimized(const float* x, float* out, int64_t n, float* workspace,
                                 cudaStream_t stream) {
  launch_single_pass<kOptimizedItemsPerThread>(x, out, n, workspace, stream);
}

}  // namespace scan
}  // namespace ops_lab
