// bindings/bind_01_execution.cpp
//
// 第 1 章的绑定：两个受控实验。
#include <torch/extension.h>

#include <algorithm>

#include <c10/cuda/CUDAStream.h>

#include "bindings/bind_helpers.h"
#include "kernels/01_execution/execution.cuh"

namespace {

cudaStream_t current_stream() { return at::cuda::getCurrentCUDAStream(); }

// 按设备规模选一个"刚好填满机器"的 grid：
// 每 SM 放满 max_threads_per_sm / block_size 个 block。
// 这是 grid-stride kernel 的常用起点。
int auto_grid_size(int block_size) {
  const ops_lab::DeviceInfo info = ops_lab::query_device();
  const int blocks_per_sm = std::max(1, info.max_threads_per_sm / block_size);
  return info.sm_count * blocks_per_sm;
}

torch::Tensor make_index_output(int64_t n) {
  return torch::empty({n}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
}

// --- 实验一：hello -------------------------------------------------------

torch::Tensor execution_hello_v1_naive(int64_t n) {
  TORCH_CHECK(n >= 0, "hello: n 必须非负，实际为 ", n);
  auto out = make_index_output(n);
  if (n == 0) {
    return out;
  }
  ops_lab::execution::hello_v1_naive(out.data_ptr<int32_t>(), n, current_stream());
  return out;
}

torch::Tensor execution_hello_v2_grid_stride(int64_t n, int64_t block_size, int64_t grid_size) {
  TORCH_CHECK(n >= 0, "hello: n 必须非负，实际为 ", n);
  TORCH_CHECK(block_size > 0, "hello: block_size 必须为正数，实际为 ", block_size);
  TORCH_CHECK(grid_size >= 0, "hello: grid_size 为 0 表示自动选择，不能为负");

  auto out = make_index_output(n);
  if (n == 0) {
    return out;
  }
  const int grid = grid_size > 0 ? static_cast<int>(grid_size)
                                 : auto_grid_size(static_cast<int>(block_size));
  ops_lab::execution::hello_v2_grid_stride(out.data_ptr<int32_t>(), n,
                                           static_cast<int>(block_size), grid, current_stream());
  return out;
}

// --- 实验二：bandwidth_probe ---------------------------------------------

torch::Tensor execution_bandwidth_probe(torch::Tensor x, int64_t variant) {
  TORCH_CHECK(x.is_cuda(), "bandwidth_probe: 输入必须在 CUDA 上");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "bandwidth_probe: 只支持 float32");

  auto xc = x.contiguous();
  auto out = torch::empty_like(xc);

  const int64_t n = xc.numel();
  if (n == 0) {
    return out;
  }

  ops_lab::execution::bandwidth_probe(xc.data_ptr<float>(), out.data_ptr<float>(), n,
                                      static_cast<int>(variant), current_stream());
  return out;
}

// 理论 occupancy（纯 host 计算，不启动 kernel）。
// 这是第 1 章实验的"自变量"读数：同一个 kernel，改 smem 就能看到占用率变化。
py::dict execution_probe_occupancy(int64_t variant, int64_t block_size) {
  const ops_lab::Occupancy oc =
      ops_lab::execution::bandwidth_probe_occupancy(static_cast<int>(variant),
                                                    static_cast<int>(block_size));
  py::dict d;
  d["variant"] = variant;
  d["block_size"] = oc.block_size;
  d["dynamic_smem_bytes"] = oc.dynamic_smem_bytes;
  d["blocks_per_sm"] = oc.blocks_per_sm;
  d["active_warps_per_sm"] = oc.active_warps_per_sm;
  d["max_warps_per_sm"] = oc.max_warps_per_sm;
  d["occupancy"] = oc.occupancy;
  return d;
}

}  // namespace

namespace ops_lab {

void bind_execution(py::module_& m) {
  bind_kernel(m, "execution_hello_v1_naive", "01_execution", "v1_naive",
              "一线程一元素、精确 grid、带边界检查；grid 有上限", &execution_hello_v1_naive);
  bind_kernel(m, "execution_hello_v2_grid_stride", "01_execution", "v2_grid_stride",
              "grid 固定、kernel 内循环步进；无 grid 上限，grid_size=0 表示自动",
              &execution_hello_v2_grid_stride);
  bind_kernel(m, "execution_bandwidth_probe", "01_execution", "probe",
              "occupancy 实验：拷贝逻辑相同，只改动态共享内存（variant 1..4）",
              &execution_bandwidth_probe);
  bind_kernel(m, "execution_probe_occupancy", "01_execution", "probe_occupancy",
              "查询某变体的理论 occupancy（纯 host 计算，不启动 kernel）",
              &execution_probe_occupancy);
}

}  // namespace ops_lab
