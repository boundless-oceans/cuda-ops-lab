// bindings/bind_03_reduction.cpp
//
// 第 3 章的 pybind11 绑定。
//
// 职责仍然是那四件事：**校验 → 分配 → 取 stream → 启动 kernel**，一行计算都不写。
//
// 与第 2 章的区别只有一处：**输出形状不再等于输入形状**。归约的输出永远是
// float32[1]，所以这里不能用 empty_like（那会得到一个同形状的张量 ——
// 值可能全对，但形状契约已经错了，而且不会有任何报错）。
//
// 空输入（n == 0）在绑定层**不做特例**：launcher 自己会写出单位元
// （sum → 0，max → -inf）并且不启动主 kernel。绑定层少一个分支，
// 就少一处"kernel 与绑定对空输入的理解不一致"的可能。
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

#include "bindings/bind_helpers.h"
#include "kernels/03_reduction/reduction.cuh"

namespace {

cudaStream_t current_stream() { return at::cuda::getCurrentCUDAStream(); }

// 归约的公共流程。
// launch 是函数指针，签名与 kernels/ 里的 launcher 一致：
//   (const float* x, float* out, int64_t n, cudaStream_t)
template <typename Launch>
torch::Tensor run_reduce(const char* op_name, torch::Tensor x, Launch launch) {
  TORCH_CHECK(x.is_cuda(), op_name, ": 输入必须在 CUDA 上");
  TORCH_CHECK(x.scalar_type() == at::kFloat, op_name, ": 目前只支持 float32");

  // 非连续输入先转连续。不这么做，data_ptr() 拿到的就不是逻辑上的连续内存，
  // 结果会**静默算错** —— 归约把整个数组加一遍，错了只会得到一个偏大/偏小的数，
  // 没有任何形状上的征兆。
  auto xc = x.contiguous();
  auto out = torch::empty({1}, xc.options());

  launch(xc.data_ptr<float>(), out.data_ptr<float>(), xc.numel(), current_stream());
  return out;
}

}  // namespace

namespace ops_lab {

void bind_reduction(py::module_& m) {
  using reduction::max_v0_uncoalesced;
  using reduction::max_v1_naive;
  using reduction::max_v2_smem_tree;
  using reduction::max_v3_warp_shuffle;
  using reduction::max_v4_single_block;
  using reduction::sum_v0_uncoalesced;
  using reduction::sum_v1_naive;
  using reduction::sum_v2_smem_tree;
  using reduction::sum_v3_warp_shuffle;
  using reduction::sum_v4_single_block;

  // ---------------------------------------------------------------------
  // v1_naive —— 基准点：合并访存 + 块内 smem 顺序寻址树 + 每 block 一次原子
  // ---------------------------------------------------------------------
  bind_kernel(m, "reduction_sum_v1_naive", "03_reduction", "v1_naive",
              "grid-stride 合并访存 + 块内 smem 顺序寻址树 + 每 block 一次 atomicAdd",
              [](torch::Tensor x) { return run_reduce("reduction_sum", x, &sum_v1_naive); });
  bind_kernel(m, "reduction_max_v1_naive", "03_reduction", "v1_naive",
              "同上，合并用 fmaxf，全局汇总用 CAS 版 atomicMax",
              [](torch::Tensor x) { return run_reduce("reduction_max", x, &max_v1_naive); });

  // ---------------------------------------------------------------------
  // v0_uncoalesced —— 反面教材：与 v1 只差索引公式那一行
  // ---------------------------------------------------------------------
  bind_kernel(m, "reduction_sum_v0_uncoalesced", "03_reduction", "v0_uncoalesced",
              "反面教材：相邻线程地址相隔 gridDim.x，一个 warp 跨 32 条 cache line",
              [](torch::Tensor x) { return run_reduce("reduction_sum", x, &sum_v0_uncoalesced); });
  bind_kernel(m, "reduction_max_v0_uncoalesced", "03_reduction", "v0_uncoalesced",
              "反面教材：访存未合并",
              [](torch::Tensor x) { return run_reduce("reduction_max", x, &max_v0_uncoalesced); });

  // ---------------------------------------------------------------------
  // v2_smem_tree —— 给 smem 加 padding；预期测不出差别
  // ---------------------------------------------------------------------
  bind_kernel(m, "reduction_sum_v2_smem_tree", "03_reduction", "v2_smem_tree",
              "smem 布局按 warp 加 padding（tid → tid + tid/32）消 bank conflict",
              [](torch::Tensor x) { return run_reduce("reduction_sum", x, &sum_v2_smem_tree); });
  bind_kernel(m, "reduction_max_v2_smem_tree", "03_reduction", "v2_smem_tree",
              "同上；顺序寻址本来就没有冲突，预期与 v1 无差别",
              [](torch::Tensor x) { return run_reduce("reduction_max", x, &max_v2_smem_tree); });

  // ---------------------------------------------------------------------
  // v3_warp_shuffle —— 块内改用 shuffle，同步 8 次 → 1 次
  // ---------------------------------------------------------------------
  bind_kernel(m, "reduction_sum_v3_warp_shuffle", "03_reduction", "v3_warp_shuffle",
              "块内前 5 级用 __shfl_down_sync，8 个 warp 和再 shuffle 一次；同步 8 次 → 1 次",
              [](torch::Tensor x) { return run_reduce("reduction_sum", x, &sum_v3_warp_shuffle); });
  bind_kernel(m, "reduction_max_v3_warp_shuffle", "03_reduction", "v3_warp_shuffle",
              "同上；预期与 v1 无差别（块内开销本就只占 0.1%）",
              [](torch::Tensor x) { return run_reduce("reduction_max", x, &max_v3_warp_shuffle); });

  // ---------------------------------------------------------------------
  // v4_single_block —— 故意回退：块内用 v3 的写法，但 grid = 1
  // ---------------------------------------------------------------------
  bind_kernel(m, "reduction_sum_v4_single_block", "03_reduction", "v4_single_block",
              "故意回退：块内沿用 v3 的 shuffle，但 grid = 1，只用 1/24 的 SM",
              [](torch::Tensor x) { return run_reduce("reduction_sum", x, &sum_v4_single_block); });
  bind_kernel(m, "reduction_max_v4_single_block", "03_reduction", "v4_single_block",
              "故意回退：grid = 1",
              [](torch::Tensor x) { return run_reduce("reduction_max", x, &max_v4_single_block); });
}

}  // namespace ops_lab
