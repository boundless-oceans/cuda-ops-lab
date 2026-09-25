// kernels/04_scan/scan_v3_single_pass.cu
//
// 第 4 章 · 阶梯第 4 级：单趟 decoupled look-back，**K = 1**（每线程一个元素）。
//
// 与 v2 相比改的是**多 block 怎么合并**：不再"读两遍 x"（12N），而是每个 block
// 处理完自己的 chunk 后把总和发布出去，再往前回看（look-back）累加已发布的和，
// 拿到 exclusive 前缀后立刻写回 —— 全局流量降到 **8N**。
//
// 这一级只改了一件事：**结构**。块内仍然是一个 chunk 256 个元素、每线程一个，
// 所以 per-chunk 的固定开销（8 次 barrier + smem 往返）和 v1/v2 一样没被摊销。
// 那件事留给 v5。
//
// 块内用的是 Kogge-Stone（不是 v2 的 Blelloch）：v1/v2 已经实测出 KS 快 25%，
// 阶梯后面的级按前面的结论走。见 scan_single_pass.cuh 的文件头。
#include "kernels/04_scan/scan.cuh"

#include "kernels/04_scan/scan_single_pass.cuh"

namespace ops_lab {
namespace scan {

void scan_inclusive_v3_single_pass(const float* x, float* out, int64_t n, float* workspace,
                                   cudaStream_t stream) {
  launch_single_pass<1>(x, out, n, workspace, stream);
}

}  // namespace scan
}  // namespace ops_lab
