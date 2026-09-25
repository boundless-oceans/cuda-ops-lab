// native/main_05_transpose.cu
//
// 第 5 章的可执行文件：不经过 Python / torch，直接跑转置六级阶梯。
//
//   ./main_05_transpose            # 默认 4096×4096
//   ./main_05_transpose 2048 200   # 边长 2048，迭代 200 次
//
//   **注意：这里的 N 是"边长"**（N×N 方阵），不是元素个数 —— Python 基准的
//   形状写的是 (4096, 4096)，两边对齐。
//
// ## 这条线做的自检：编号矩阵 + **刻意挑一个不对齐的形状**
//
// 输入用 `in[r][c] = r*cols + c`（元素值就是它的输入下标），于是
// `out[c][r]` 必须**精确**等于 `r*cols + c` —— 任何"搬错位置"都无处可藏。
//
// 自检里特意混进 `(37, 53)`、`(128, 130)` 这种**行数/列数不是 4 的倍数**的形状：
// float4 快路径在这种形状上必须退回标量，否则会得到 `misaligned address`。
// （真事：第一版漏了这个判据，测试里 10 个用例全红，而且脏上下文让
// 连没有 float4 的 v0 也报同一个错，非常误导。）
#include <cstdio>
#include <functional>
#include <vector>

#include "native/bench_main.h"

#include "kernels/05_transpose/transpose.cuh"

namespace {

using TransposeCall = std::function<void(const float*, float*, int, int)>;

struct Variant {
  const char* label;
  TransposeCall call;
};

const Variant kVariants[] = {
    {"v0_naive", [](const float* i, float* o, int r, int c) {
       ops_lab::transpose::transpose_v0_naive(i, o, r, c, 0);
     }},
    {"v1_swapped", [](const float* i, float* o, int r, int c) {
       ops_lab::transpose::transpose_v1_swapped(i, o, r, c, 0);
     }},
    {"v2_smem_tiled", [](const float* i, float* o, int r, int c) {
       ops_lab::transpose::transpose_v2_smem_tiled(i, o, r, c, 0);
     }},
    {"v3_padded", [](const float* i, float* o, int r, int c) {
       ops_lab::transpose::transpose_v3_padded(i, o, r, c, 0);
     }},
    {"v4_vectorized", [](const float* i, float* o, int r, int c) {
       ops_lab::transpose::transpose_v4_vectorized(i, o, r, c, 0);
     }},
    {"v5_wide_tile", [](const float* i, float* o, int r, int c) {
       ops_lab::transpose::transpose_v5_wide_tile(i, o, r, c, 0);
     }},
};

// 编号矩阵：in[r][c] = r*cols + c
std::vector<float> numbered(int rows, int cols) {
  std::vector<float> v(static_cast<size_t>(rows) * cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      v[static_cast<size_t>(r) * cols + c] = static_cast<float>(r * cols + c);
    }
  }
  return v;
}

// 返回第一个不合格的输出下标（全都合格返回 -1）
int64_t check_numbered(const std::vector<float>& in, int rows, int cols,
                       const std::vector<float>& out) {
  for (int c = 0; c < cols; ++c) {
    for (int r = 0; r < rows; ++r) {
      const size_t oi = static_cast<size_t>(c) * rows + r;
      if (out[oi] != in[static_cast<size_t>(r) * cols + c]) {
        return static_cast<int64_t>(oi);
      }
    }
  }
  return -1;
}

}  // namespace

namespace {

int run(int argc, char** argv) {
  using namespace ops_lab::native;

  const NativeOptions opt = parse_options(argc, argv);
  if (require_device()) {
    return 2;
  }

  // ---------------------------------------------------------- ① 结果自检
  // (37,53) 行与列都不是 4 的倍数 → float4 必须退回标量
  // (64,64)  两个方向都是 4 的倍数 → 走 float4 快路径
  // (128,130) 行是、列不是 → 只有一侧对齐
  const int kSelfCheck[][2] = {{37, 53}, {64, 64}, {128, 130}};

  std::printf("\n=== 05_transpose / 结果自检（编号矩阵，判据 = 逐位相等）===\n");
  std::printf("输入 in[r][c] = r*cols + c，所以 out[c][r] 必须精确等于它。\n");
  std::printf("形状里特意混了行/列不是 4 的倍数的情形 —— float4 快路径必须能退回标量。\n\n");

  bool all_ok = true;
  for (const auto& rc : kSelfCheck) {
    const int rows = rc[0];
    const int cols = rc[1];
    const std::vector<float> in_host = numbered(rows, cols);
    DeviceBuffer d_in(static_cast<int64_t>(rows) * cols);
    DeviceBuffer d_out(static_cast<int64_t>(rows) * cols);
    upload_pattern(d_in, in_host);

    std::vector<float> out_host(static_cast<size_t>(rows) * cols);
    for (const Variant& v : kVariants) {
      v.call(d_in.get(), d_out.get(), rows, cols);
      OPSLAB_CUDA_CHECK(cudaMemcpy(out_host.data(), d_out.get(),
                                   sizeof(float) * out_host.size(), cudaMemcpyDeviceToHost));
      const int64_t bad = check_numbered(in_host, rows, cols, out_host);
      if (bad < 0) {
        std::printf("  %-18s %d×%d  OK\n", v.label, rows, cols);
      } else {
        std::printf("  %-18s %d×%d  第 %lld 个输出不对：实际 %.0f，期望 %.0f  <<< 不一致\n",
                    v.label, rows, cols, static_cast<long long>(bad),
                    static_cast<double>(out_host[static_cast<size_t>(bad)]),
                    static_cast<double>(in_host[(bad % rows) * cols + (bad / rows)]));
        all_ok = false;
      }
    }
  }
  if (!all_ok) {
    std::fprintf(stderr, "\n[警告] 有变体的结果与编号矩阵不一致。\n");
    std::fprintf(stderr, "        计时仍会打印，但**不要**拿这些数字做性能分析 ——\n");
    std::fprintf(stderr, "        而且 misaligned/越界这类错误会污染上下文，\n");
    std::fprintf(stderr, "        之后所有变体都会跟着报错（真发生过）。\n");
  }

  // ------------------------------------------------ ② 稳态预热 + 计时表
  const int side = static_cast<int>(opt.n);  // N 是边长
  const int64_t elems = static_cast<int64_t>(side) * side;
  DeviceBuffer x(elems);
  DeviceBuffer out(elems);
  fill_pattern(x, /*seed=*/20260914);

  steady_state_warmup([&] { ops_lab::transpose::transpose_v3_padded(
                                x.get(), out.get(), side, side, 0); },
                      2.0);

  const double peak = peak_gbps();
  const int64_t bytes = 8 * elems;  // 读 4MN + 写 4MN
  if (!print_header("05_transpose / transpose（读 + 写 = 8MN）", opt, bytes)) {
    return 2;
  }
  for (const Variant& v : kVariants) {
    bench_row(v.label, bytes,
              [&] { v.call(x.get(), out.get(), side, side); }, opt, peak);
  }

  std::printf("\n提示：把本可执行文件直接喂给 ncu 就能拿到硬件计数器。\n");
  std::printf("      最值得看的是 v0 与 v1 —— 它们只差\"跨步在读还是在写\"：\n");
  std::printf("      ncu -k \"regex:transpose_v0_kernel\" --launch-count 1 --set full "
              "./main_05_transpose %lld 10\n",
              static_cast<long long>(opt.n));
  std::printf("      预期 dram__bytes_read.sum 在 v0 上明显更高（部分 sector 写要 RMW）。\n");
  return all_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) { return ops_lab::native::run_guarded(run, argc, argv); }
