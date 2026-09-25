// kernels/04_scan/scan_v2_blelloch.cu
//
// 第 4 章 · 阶梯第 3 级：块内换成 work-efficient 的 Blelloch（上扫 + 下扫）。
//
// 与 v1 只差一个模板参数（`KoggeStoneScanPolicy` → `BlellochScanPolicy`）。
// work 从每元素 O(log B) 降到 O(1) 摊销，代价是级数从 8 涨到 16。
//
// 预期：块内 work（约 0.028 ms）远低于内存时间，**与 v1 几乎一样快**
// （两者都被 12N 的内存下限 393 µs 卡住）。
//
// 这一级和第 3 章的 v2/v3 有点像（都是"改造块内"，预期测不出差别），
// 但性质不同：第 3 章是"块内本来就不值钱"，这里是"块内已经便宜到被掩盖"。
// 如果实测 v2 明显快于 v1，说明块内 work 尚未完全被掩盖，那也是一个真结论。
#include "kernels/04_scan/scan.cuh"

#include "kernels/04_scan/scan_three_pass.cuh"

namespace ops_lab {
namespace scan {

void scan_inclusive_v2_blelloch(const float* x, float* out, int64_t n, float* workspace,
                                cudaStream_t stream) {
  run_three_pass<BlellochScanPolicy>(x, out, n, workspace, stream);
}

}  // namespace scan
}  // namespace ops_lab
