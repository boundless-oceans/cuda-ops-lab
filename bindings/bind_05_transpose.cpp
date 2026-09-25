// bindings/bind_05_transpose.cpp
//
// 第 5 章的 pybind11 绑定。
//
// 职责照旧：**校验 → 分配 → 取 stream → 启动 kernel**。
//
// 与前面几章的差别：
//   1. **输入必须是 2 维**（`x.dim() == 2`）。转置是二维算子，比"展平处理"
//      更清楚 —— 展平一个 3 维张量再转置没有明确语义。
//   2. **输出形状是 (cols, rows)**，与输入不同（用 `torch::empty({cols, rows})`）。
//      元数据里用 `out_shape` 声明，第 3 章加的那个字段正好在这里第二次派上用场。
//   3. **用 int 传尺寸**：kernel 里的下标是 `c*rows + r`，用 int64 会多占寄存器。
//      代价是大矩阵有溢出风险，所以这里显式拦住（2^31 个元素 = 单精度 8 GB，
//      实际上先撞到显存）。
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

#include "bindings/bind_helpers.h"
#include "kernels/05_transpose/transpose.cuh"

namespace {

cudaStream_t current_stream() { return at::cuda::getCurrentCUDAStream(); }

// launch 签名：(const float* in, float* out, int rows, int cols)
template <typename Launch>
torch::Tensor run_transpose(const char* op_name, torch::Tensor x, Launch launch) {
  TORCH_CHECK(x.is_cuda(), op_name, ": 输入必须在 CUDA 上");
  TORCH_CHECK(x.scalar_type() == at::kFloat, op_name, ": 目前只支持 float32");
  TORCH_CHECK(x.dim() == 2, op_name, ": 只支持 2 维输入（转置是二维算子），实际 ", x.dim(),
              " 维");

  // 非连续输入先转连续。转置算子里这一步特别容易踩：`x.t()` 本身是**视图**，
  // 不转连续就取 data_ptr() 会把"逻辑上的转置"当成"内存里的转置"，结果又转回去了。
  auto xc = x.contiguous();
  const int64_t rows64 = xc.size(0);
  const int64_t cols64 = xc.size(1);
  TORCH_CHECK(rows64 * cols64 <= 2147483647LL, op_name,
              ": 元素数超过 int32 上限，kernel 里的下标会溢出（rows*cols = ",
              rows64 * cols64, "）");

  auto out = torch::empty({cols64, rows64}, xc.options());
  if (rows64 == 0 || cols64 == 0) {
    return out;  // 空矩阵：输出已经是对的形状，不需要启动 kernel
  }
  launch(xc.data_ptr<float>(), out.data_ptr<float>(), static_cast<int>(rows64),
         static_cast<int>(cols64));
  return out;
}

}  // namespace

namespace ops_lab {

void bind_transpose(py::module_& m) {
  using transpose::transpose_v0_naive;
  using transpose::transpose_v1_swapped;
  using transpose::transpose_v2_smem_tiled;
  using transpose::transpose_v3_padded;
  using transpose::transpose_v4_vectorized;
  using transpose::transpose_v5_wide_tile;

  // ---------------------------------------------------------------------
  // v0 / v1 —— 两个反面教材（只差相邻线程沿哪个下标走）
  // ---------------------------------------------------------------------
  bind_kernel(m, "transpose_v0_naive", "05_transpose", "v0_naive",
              "反面教材：读合并、写跨步（相邻线程隔 rows 个元素）",
              [](torch::Tensor x) {
                return run_transpose("transpose", x, [](const float* p, float* o, int r, int c) {
                  transpose_v0_naive(p, o, r, c, current_stream());
                });
              });
  bind_kernel(m, "transpose_v1_swapped", "05_transpose", "v1_swapped",
              "反面教材：读跨步、写合并；与 v0 只差交换 (r,c) 的取法",
              [](torch::Tensor x) {
                return run_transpose("transpose", x, [](const float* p, float* o, int r, int c) {
                  transpose_v1_swapped(p, o, r, c, current_stream());
                });
              });

  // ---------------------------------------------------------------------
  // v2 / v3 —— smem 瓦片，与 v2 只差共享内存的行宽
  // ---------------------------------------------------------------------
  bind_kernel(m, "transpose_v2_smem_tiled", "05_transpose", "v2_smem_tiled",
              "32×32 共享内存瓦片：两边都合并，但列访问有 32 路 bank conflict",
              [](torch::Tensor x) {
                return run_transpose("transpose", x, [](const float* p, float* o, int r, int c) {
                  transpose_v2_smem_tiled(p, o, r, c, current_stream());
                });
              });
  bind_kernel(m, "transpose_v3_padded", "05_transpose", "v3_padded",
              "与 v2 只差共享内存行宽 32→33（消掉 32 路 bank conflict）",
              [](torch::Tensor x) {
                return run_transpose("transpose", x, [](const float* p, float* o, int r, int c) {
                  transpose_v3_padded(p, o, r, c, current_stream());
                });
              });

  // ---------------------------------------------------------------------
  // v4 / v5 —— 向量化，与 v4 只差瓦片宽度
  // ---------------------------------------------------------------------
  bind_kernel(m, "transpose_v4_vectorized", "05_transpose", "v4_vectorized",
              "与 v3 只差访存宽度：读写两侧都用 float4（线程映射随之改变）",
              [](torch::Tensor x) {
                return run_transpose("transpose", x, [](const float* p, float* o, int r, int c) {
                  transpose_v4_vectorized(p, o, r, c, current_stream());
                });
              });
  bind_kernel(m, "transpose_v5_wide_tile", "05_transpose", "v5_wide_tile",
              "与 v4 只差瓦片宽度 32→64：每次连续访问 128 B → 256 B",
              [](torch::Tensor x) {
                return run_transpose("transpose", x, [](const float* p, float* o, int r, int c) {
                  transpose_v5_wide_tile(p, o, r, c, current_stream());
                });
              });
}

}  // namespace ops_lab
