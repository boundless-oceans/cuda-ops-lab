// kernels/04_scan/scan_v0_naive.cu
//
// 第 4 章 · 阶梯第 1 级：反面教材 —— 块内用 O(B²) 的朴素扫描。
//
// 与 v1 只差一个模板参数（`NaiveScanPolicy` ↔ `KoggeStoneScanPolicy`）：
// 三段式的流程、载入、加前缀、写回全都相同。
//
// 为什么这是"反面教材"而不是"能跑的版本"：线程 t 要把自己前面 t 个元素加一遍，
// 于是每个 chunk 的 work 是 B(B+1)/2 次加法 —— 每元素 O(B)，全数组 O(N·B)。
// 第 3 章的 reduction 每元素只有 O(1) 次合并，所以块内开销可以忽略；
// **这里不行**，这笔账在 scan.cuh 里算过：块内 work 会超过内存时间。
//
// 预期：~25~30% 峰值（瓶颈是 smem 吞吐，不是显存带宽）。
#include "kernels/04_scan/scan.cuh"

#include "kernels/04_scan/scan_three_pass.cuh"

namespace ops_lab {
namespace scan {

void scan_inclusive_v0_naive(const float* x, float* out, int64_t n, float* workspace,
                             cudaStream_t stream) {
  run_three_pass<NaiveScanPolicy>(x, out, n, workspace, stream);
}

}  // namespace scan
}  // namespace ops_lab
