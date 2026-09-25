// kernels/04_scan/scan_v1_kogge_stone.cu
//
// 第 4 章 · 阶梯第 2 级：块内换成 Kogge-Stone（Hillis-Steele）。
//
// 与 v0 只差一个模板参数（`NaiveScanPolicy` → `KoggeStoneScanPolicy`）。
// work 从每元素 O(B) 降到 O(log B)：8 级、每级每个线程 1 次读 + 1 次写。
//
// 预期：块内 work（约 0.10 ms）已经低于内存时间（262 µs），所以应该回到
// **内存受限**：时间≈12N 的下限 393 µs，也就是按实际流量算的 ~90% 峰值。
#include "kernels/04_scan/scan.cuh"

#include "kernels/04_scan/scan_three_pass.cuh"

namespace ops_lab {
namespace scan {

void scan_inclusive_v1_kogge_stone(const float* x, float* out, int64_t n, float* workspace,
                                   cudaStream_t stream) {
  run_three_pass<KoggeStoneScanPolicy>(x, out, n, workspace, stream);
}

}  // namespace scan
}  // namespace ops_lab
