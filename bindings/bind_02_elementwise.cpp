// bindings/bind_02_elementwise.cpp
//
// 第 2 章的 pybind11 绑定。
//
// 绑定层的职责只有四件事：**校验 → 分配 → 取 stream → 启动 kernel**。
// 一行计算逻辑都不应该有 —— 计算永远在 kernels/ 里，这样同一份 kernel
// 才能被 native/ 的纯 CUDA 可执行文件复用（design §5.1）。
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

#include "bindings/bind_helpers.h"
#include "kernels/02_elementwise/elementwise.cuh"

namespace {

// elementwise add 的 v1_naive 变体（一线程一元素、已合并访存）。
torch::Tensor elementwise_add_v1_naive(torch::Tensor x, torch::Tensor y) {
  TORCH_CHECK(x.is_cuda() && y.is_cuda(), "elementwise_add: 两个输入都必须在 CUDA 上");
  TORCH_CHECK(x.scalar_type() == at::kFloat && y.scalar_type() == at::kFloat,
              "elementwise_add: 目前只支持 float32");
  TORCH_CHECK(x.sizes() == y.sizes(), "elementwise_add: 两个输入的 shape 必须一致，实际为 ",
              x.sizes(), " 与 ", y.sizes());

  // 非连续输入先转连续。不做这一步，data_ptr() 拿到的就不是逻辑上的连续内存，
  // 结果会静默算错 —— 这类 bug 在比赛里非常常见。
  auto xc = x.contiguous();
  auto yc = y.contiguous();
  auto out = torch::empty_like(xc);

  const int64_t n = xc.numel();
  if (n == 0) {
    // 空张量必须能跑通，不能去启动一个 grid == 0 的 kernel。
    return out;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ops_lab::elementwise::add_v1_naive(xc.data_ptr<float>(), yc.data_ptr<float>(),
                                     out.data_ptr<float>(), n, stream);
  return out;
}

}  // namespace

namespace ops_lab {

void bind_elementwise(py::module_& m) {
  bind_kernel(m, "elementwise_add_v1_naive", "02_elementwise", "v1_naive",
              "一线程一元素，连续访问（已合并访存）；无 grid-stride，grid 有上限",
              &elementwise_add_v1_naive);
}

}  // namespace ops_lab
