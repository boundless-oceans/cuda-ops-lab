// bindings/bind_02_elementwise.cpp
//
// 第 2 章的 pybind11 绑定。
//
// 绑定层的职责只有四件事：**校验 → 分配 → 取 stream → 启动 kernel**。
// 一行计算逻辑都不应该有 —— 计算永远在 kernels/ 里，这样同一份 kernel
// 才能被 native/ 的纯 CUDA 可执行文件复用。
//
// 三个算子（add / relu / sigmoid）× 每一级变体 = 十几个导出函数。
// 它们的绑定代码是**完全相同**的，只有最后启动哪个 kernel 不同，
// 所以公共部分抽成 run_binary / run_unary 两个模板 ——
// 否则这一层会变成几百行复制粘贴。
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

#include "bindings/bind_helpers.h"
#include "kernels/02_elementwise/elementwise.cuh"

namespace {

cudaStream_t current_stream() { return at::cuda::getCurrentCUDAStream(); }

// 二目算子（add）的公共流程。
// launch 是函数指针，签名与 kernels/ 里的 launcher 一致：
//   (const float* x, const float* y, float* out, int64_t n, cudaStream_t)
template <typename Launch>
torch::Tensor run_binary(const char* op_name, torch::Tensor x, torch::Tensor y, Launch launch) {
  TORCH_CHECK(x.is_cuda() && y.is_cuda(), op_name, ": 两个输入都必须在 CUDA 上");
  TORCH_CHECK(x.scalar_type() == at::kFloat && y.scalar_type() == at::kFloat,
              op_name, ": 目前只支持 float32");
  TORCH_CHECK(x.sizes() == y.sizes(), op_name, ": 两个输入的 shape 必须一致，实际为 ",
              x.sizes(), " 与 ", y.sizes());

  // 非连续输入先转连续。不做这一步，data_ptr() 拿到的就不是逻辑上的连续内存，
  // 结果会**静默算错** —— 这类 bug 在比赛里非常常见。
  auto xc = x.contiguous();
  auto yc = y.contiguous();
  auto out = torch::empty_like(xc);

  const int64_t n = xc.numel();
  if (n == 0) {
    return out;  // 空张量必须能跑通，不能去启动一个 grid == 0 的 kernel
  }
  launch(xc.data_ptr<float>(), yc.data_ptr<float>(), out.data_ptr<float>(), n, current_stream());
  return out;
}

// 一目算子（relu / sigmoid）的公共流程。
template <typename Launch>
torch::Tensor run_unary(const char* op_name, torch::Tensor x, Launch launch) {
  TORCH_CHECK(x.is_cuda(), op_name, ": 输入必须在 CUDA 上");
  TORCH_CHECK(x.scalar_type() == at::kFloat, op_name, ": 目前只支持 float32");

  auto xc = x.contiguous();
  auto out = torch::empty_like(xc);

  const int64_t n = xc.numel();
  if (n == 0) {
    return out;
  }
  launch(xc.data_ptr<float>(), out.data_ptr<float>(), n, current_stream());
  return out;
}

}  // namespace

namespace ops_lab {

void bind_elementwise(py::module_& m) {
  using elementwise::add_v0_uncoalesced;
  using elementwise::add_v1_naive;
  using elementwise::add_v2_unroll4;
  using elementwise::add_v3_grid_stride;
  using elementwise::add_v4_vectorized;
  using elementwise::relu_v0_uncoalesced;
  using elementwise::relu_v1_naive;
  using elementwise::relu_v2_unroll4;
  using elementwise::relu_v3_grid_stride;
  using elementwise::relu_v4_vectorized;
  using elementwise::sigmoid_v0_uncoalesced;
  using elementwise::sigmoid_v1_naive;
  using elementwise::sigmoid_v2_unroll4;
  using elementwise::sigmoid_v3_grid_stride;
  using elementwise::sigmoid_v4_vectorized;

  // ---------------------------------------------------------------------
  // v1_naive —— 基准点：已合并访存，但每线程只有 1 个访存在飞
  // ---------------------------------------------------------------------
  bind_kernel(m, "elementwise_add_v1_naive", "02_elementwise", "v1_naive",
              "一线程一元素、连续访问（已合并访存）；无 grid-stride，grid 有上限",
              [](torch::Tensor x, torch::Tensor y) {
                return run_binary("elementwise_add", x, y, &add_v1_naive);
              });
  bind_kernel(m, "elementwise_relu_v1_naive", "02_elementwise", "v1_naive",
              "一线程一元素、连续访问（已合并访存）",
              [](torch::Tensor x) { return run_unary("elementwise_relu", x, &relu_v1_naive); });
  bind_kernel(m, "elementwise_sigmoid_v1_naive", "02_elementwise", "v1_naive",
              "一线程一元素、连续访问；带 exp，用来观察算术强度上升后是否仍受带宽限制",
              [](torch::Tensor x) {
                return run_unary("elementwise_sigmoid", x, &sigmoid_v1_naive);
              });

  // ---------------------------------------------------------------------
  // v0_uncoalesced —— 反面教材：与 v1 只差索引公式那一行
  // ---------------------------------------------------------------------
  bind_kernel(m, "elementwise_add_v0_uncoalesced", "02_elementwise", "v0_uncoalesced",
              "反面教材：相邻线程地址相隔 gridDim.x，一个 warp 跨 32 条 cache line",
              [](torch::Tensor x, torch::Tensor y) {
                return run_binary("elementwise_add", x, y, &add_v0_uncoalesced);
              });
  bind_kernel(m, "elementwise_relu_v0_uncoalesced", "02_elementwise", "v0_uncoalesced",
              "反面教材：访存未合并",
              [](torch::Tensor x) { return run_unary("elementwise_relu", x, &relu_v0_uncoalesced); });
  bind_kernel(m, "elementwise_sigmoid_v0_uncoalesced", "02_elementwise", "v0_uncoalesced",
              "反面教材：访存未合并", [](torch::Tensor x) {
                return run_unary("elementwise_sigmoid", x, &sigmoid_v0_uncoalesced);
              });

  // ---------------------------------------------------------------------
  // v2_unroll4 —— 每线程 4 个元素，靠 ILP 提高在飞访存请求数
  // ---------------------------------------------------------------------
  bind_kernel(m, "elementwise_add_v2_unroll4", "02_elementwise", "v2_unroll4",
              "每线程 4 个元素：先全部 load 再统一 store，用 ILP 换并发度",
              [](torch::Tensor x, torch::Tensor y) {
                return run_binary("elementwise_add", x, y, &add_v2_unroll4);
              });
  bind_kernel(m, "elementwise_relu_v2_unroll4", "02_elementwise", "v2_unroll4",
              "每线程 4 个元素，靠 ILP 提高并发度",
              [](torch::Tensor x) { return run_unary("elementwise_relu", x, &relu_v2_unroll4); });
  bind_kernel(m, "elementwise_sigmoid_v2_unroll4", "02_elementwise", "v2_unroll4",
              "每线程 4 个元素，靠 ILP 提高并发度",
              [](torch::Tensor x) { return run_unary("elementwise_sigmoid", x, &sigmoid_v2_unroll4); });

  // ---------------------------------------------------------------------
  // v3_grid_stride —— grid 固定成"刚好填满机器"，循环步进覆盖任意大的 n
  // ---------------------------------------------------------------------
  bind_kernel(m, "elementwise_add_v3_grid_stride", "02_elementwise", "v3_grid_stride",
              "grid 固定为 SM 数 × 每 SM 块数，循环步进；无 grid 上限",
              [](torch::Tensor x, torch::Tensor y) {
                return run_binary("elementwise_add", x, y, &add_v3_grid_stride);
              });
  bind_kernel(m, "elementwise_relu_v3_grid_stride", "02_elementwise", "v3_grid_stride",
              "grid 固定 + 循环步进；无 grid 上限",
              [](torch::Tensor x) { return run_unary("elementwise_relu", x, &relu_v3_grid_stride); });
  bind_kernel(m, "elementwise_sigmoid_v3_grid_stride", "02_elementwise", "v3_grid_stride",
              "grid 固定 + 循环步进；无 grid 上限", [](torch::Tensor x) {
                return run_unary("elementwise_sigmoid", x, &sigmoid_v3_grid_stride);
              });

  // ---------------------------------------------------------------------
  // v4_vectorized —— float4，访存指令数降 4 倍；尾部走标量路径
  // ---------------------------------------------------------------------
  bind_kernel(m, "elementwise_add_v4_vectorized", "02_elementwise", "v4_vectorized",
              "float4 向量化，访存指令数降 4 倍；非 4 倍数尾部走标量路径",
              [](torch::Tensor x, torch::Tensor y) {
                return run_binary("elementwise_add", x, y, &add_v4_vectorized);
              });
  bind_kernel(m, "elementwise_relu_v4_vectorized", "02_elementwise", "v4_vectorized",
              "float4 向量化；非对齐时退回标量路径",
              [](torch::Tensor x) { return run_unary("elementwise_relu", x, &relu_v4_vectorized); });
  bind_kernel(m, "elementwise_sigmoid_v4_vectorized", "02_elementwise", "v4_vectorized",
              "float4 向量化；非对齐时退回标量路径", [](torch::Tensor x) {
                return run_unary("elementwise_sigmoid", x, &sigmoid_v4_vectorized);
              });
}

}  // namespace ops_lab
