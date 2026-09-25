// bindings/bind_04_scan.cpp
//
// 第 4 章的 pybind11 绑定。
//
// 职责还是那四件事：**校验 → 分配 → 取 stream → 启动 kernel**。
//
// 与第 3 章的两处差别：
//
//   1. **输出形状与输入一致**（scan 是一对一映射），所以用 `empty_like` 而不是
//      `empty({1})`。（第 3 章是反过来的：那里输出固定 (1,)，所以专门加了
//      `out_shape` 元数据字段。）
//   2. **需要 workspace**。三段式要一个"每 chunk 一个前缀"的数组，单趟实现要
//      aggregate/inclusive/status 三段。第 3 章曾经讨论过 workspace 但最后没用上
//      （那里的多 block 合并是免费的原子操作）；scan 的合并第一次要付真实代价，
//      所以这里必须分配。
//
// 空输入（n == 0）直接返回空输出、不启动 kernel —— 注意这和第 3 章不同：
// 归约的空输入必须写出单位元（sum→0 / max→-inf），所以那里不能提前返回；
// scan 的输出是空的，没有"单位元输出"这回事。
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

#include "bindings/bind_helpers.h"
#include "kernels/04_scan/scan.cuh"

namespace {

cudaStream_t current_stream() { return at::cuda::getCurrentCUDAStream(); }

// scan 的公共流程。
//   launch 签名：(const float* x, float* out, int64_t n, float* workspace)
//   needs_workspace = false 时传 nullptr（v4 单 block 变体不需要）
template <typename Launch>
torch::Tensor run_scan(const char* op_name, torch::Tensor x, bool needs_workspace,
                       Launch launch) {
  TORCH_CHECK(x.is_cuda(), op_name, ": 输入必须在 CUDA 上");
  TORCH_CHECK(x.scalar_type() == at::kFloat, op_name, ": 目前只支持 float32");

  // 非连续输入先转连续。不这么做，data_ptr() 拿到的就不是逻辑上的连续内存，
  // 结果会**静默算错** —— 而且 scan 的错法是"后面所有元素都偏"，看起来很像精度问题，
  // 特别难查。
  auto xc = x.contiguous();
  auto out = torch::empty_like(xc);  // scan 输出与输入同形状
  const int64_t n = xc.numel();
  if (n == 0) {
    return out;  // 空输入：输出也是空的，不需要启动 kernel
  }

  // workspace 由 torch 的缓存分配器管理。注意它必须在 kernel 跑完之前一直有效 ——
  // 这里靠的是"同一 stream 上的分配/释放是流序的"这条保证（本仓库是单 stream 用法）。
  torch::Tensor ws;
  float* ws_ptr = nullptr;
  if (needs_workspace) {
    ws = torch::empty({ops_lab::scan::workspace_floats(n)}, xc.options());
    ws_ptr = ws.data_ptr<float>();
  }

  launch(xc.data_ptr<float>(), out.data_ptr<float>(), n, ws_ptr);
  return out;
}

}  // namespace

namespace ops_lab {

void bind_scan(py::module_& m) {
  using scan::scan_inclusive_v0_naive;
  using scan::scan_inclusive_v1_kogge_stone;
  using scan::scan_inclusive_v2_blelloch;
  using scan::scan_inclusive_v3_single_pass;
  using scan::scan_inclusive_v4_optimized;
  using scan::scan_inclusive_v5_single_block;

  // ---------------------------------------------------------------------
  // v0_naive —— 反面教材：块内 O(B²)
  // ---------------------------------------------------------------------
  bind_kernel(m, "scan_inclusive_v0_naive", "04_scan", "v0_naive",
              "反面教材：三段式 + 块内朴素扫描（每元素 O(B)，work 是内存时间的 2.7 倍）",
              [](torch::Tensor x) {
                return run_scan("scan_inclusive", x, true,
                                [](const float* px, float* po, int64_t n, float* ws) {
                                  scan_inclusive_v0_naive(px, po, n, ws, current_stream());
                                });
              });

  // ---------------------------------------------------------------------
  // v1_kogge_stone —— 块内 work 降到 O(N log B)
  // ---------------------------------------------------------------------
  bind_kernel(m, "scan_inclusive_v1_kogge_stone", "04_scan", "v1_kogge_stone",
              "三段式 + 块内 Kogge-Stone（8 级）；估计回到内存受限",
              [](torch::Tensor x) {
                return run_scan("scan_inclusive", x, true,
                                [](const float* px, float* po, int64_t n, float* ws) {
                                  scan_inclusive_v1_kogge_stone(px, po, n, ws, current_stream());
                                });
              });

  // ---------------------------------------------------------------------
  // v2_blelloch —— work-efficient（每元素 O(1) 摊销）
  // ---------------------------------------------------------------------
  bind_kernel(m, "scan_inclusive_v2_blelloch", "04_scan", "v2_blelloch",
              "三段式 + 块内 Blelloch 上扫/下扫；work 最少但级数翻倍",
              [](torch::Tensor x) {
                return run_scan("scan_inclusive", x, true,
                                [](const float* px, float* po, int64_t n, float* ws) {
                                  scan_inclusive_v2_blelloch(px, po, n, ws, current_stream());
                                });
              });

  // ---------------------------------------------------------------------
  // v3_single_pass —— 单趟 decoupled look-back（8N 而不是 12N）
  // ---------------------------------------------------------------------
  bind_kernel(m, "scan_inclusive_v3_single_pass", "04_scan", "v3_single_pass",
              "单趟 decoupled look-back：只读一遍输入（8N），靠 block 间发布/自旋建立顺序",
              [](torch::Tensor x) {
                return run_scan("scan_inclusive", x, true,
                                [](const float* px, float* po, int64_t n, float* ws) {
                                  scan_inclusive_v3_single_pass(px, po, n, ws, current_stream());
                                });
              });

  // ---------------------------------------------------------------------
  // v4_optimized —— 单趟 look-back + 每线程 8 个元素（本章的答案）
  // ---------------------------------------------------------------------
  bind_kernel(m, "scan_inclusive_v4_optimized", "04_scan", "v4_optimized",
              "单趟 look-back + 每线程 8 个元素（chunk 2048）：摊销 per-chunk 开销，8N 流量",
              [](torch::Tensor x) {
                return run_scan("scan_inclusive", x, true,
                                [](const float* px, float* po, int64_t n, float* ws) {
                                  scan_inclusive_v4_optimized(px, po, n, ws, current_stream());
                                });
              });

  // ---------------------------------------------------------------------
  // v5_single_block —— 故意回退：grid = 1
  // ---------------------------------------------------------------------
  bind_kernel(m, "scan_inclusive_v5_single_block", "04_scan", "v5_single_block",
              "故意回退：grid = 1，一个 block 顺序走完所有 chunk（不需要 workspace）",
              [](torch::Tensor x) {
                return run_scan("scan_inclusive", x, false,
                                [](const float* px, float* po, int64_t n, float* /*ws*/) {
                                  scan_inclusive_v5_single_block(px, po, n, current_stream());
                                });
              });
}

}  // namespace ops_lab
