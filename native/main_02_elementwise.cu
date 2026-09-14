// native/main_02_elementwise.cu
//
// 第 2 章的可执行文件：不经过 Python / torch，直接跑 elementwise 五级阶梯。
//
//   ./main_02_elementwise                  # 默认 N = 2^24
//   ./main_02_elementwise 1048576 200      # 指定 N 与迭代次数
//   ncu --set full ./main_02_elementwise   # 直接喂给性能分析工具
//
// 数字应当与 `python bench/run_bench.py --chapter elementwise` 互相印证 ——
// 两边用的是同一份 kernels/ 和同一套计时/峰值公式。
#include <cstdio>

#include "native/bench_main.h"

#include "kernels/02_elementwise/elementwise.cuh"

namespace {

using BinaryFn = void (*)(const float*, const float*, float*, int64_t, cudaStream_t);
using UnaryFn = void (*)(const float*, float*, int64_t, cudaStream_t);

struct BinaryVariant {
  const char* label;
  BinaryFn fn;
};

struct UnaryVariant {
  const char* label;
  UnaryFn fn;
};

const BinaryVariant kAddVariants[] = {
    {"v0_uncoalesced", ops_lab::elementwise::add_v0_uncoalesced},
    {"v1_naive", ops_lab::elementwise::add_v1_naive},
    {"v2_unroll4", ops_lab::elementwise::add_v2_unroll4},
    {"v3_grid_stride", ops_lab::elementwise::add_v3_grid_stride},
    {"v4_vectorized", ops_lab::elementwise::add_v4_vectorized},
};

const UnaryVariant kReluVariants[] = {
    {"v0_uncoalesced", ops_lab::elementwise::relu_v0_uncoalesced},
    {"v1_naive", ops_lab::elementwise::relu_v1_naive},
    {"v2_unroll4", ops_lab::elementwise::relu_v2_unroll4},
    {"v3_grid_stride", ops_lab::elementwise::relu_v3_grid_stride},
    {"v4_vectorized", ops_lab::elementwise::relu_v4_vectorized},
};

const UnaryVariant kSigmoidVariants[] = {
    {"v0_uncoalesced", ops_lab::elementwise::sigmoid_v0_uncoalesced},
    {"v1_naive", ops_lab::elementwise::sigmoid_v1_naive},
    {"v2_unroll4", ops_lab::elementwise::sigmoid_v2_unroll4},
    {"v3_grid_stride", ops_lab::elementwise::sigmoid_v3_grid_stride},
    {"v4_vectorized", ops_lab::elementwise::sigmoid_v4_vectorized},
};

// 一目算子的三段结构完全一样，抽出来免得抄三遍
void bench_unary_table(const char* title, const UnaryVariant* variants, int count,
                       float* x, float* out, const ops_lab::native::NativeOptions& opt,
                       double peak) {
  const int64_t bytes = opt.n * 8;  // 读 4N + 写 4N
  if (!ops_lab::native::print_header(title, opt, bytes)) {
    return;
  }
  for (int i = 0; i < count; ++i) {
    const UnaryVariant& v = variants[i];
    ops_lab::native::bench_row(
        v.label, bytes, [&] { v.fn(x, out, opt.n, 0); }, opt, peak);
  }
}

}  // namespace

namespace {

int run(int argc, char** argv) {
  using namespace ops_lab::native;

  const NativeOptions opt = parse_options(argc, argv);

  // ------ add：读 x + 读 y + 写 out = 12 字节/元素 ------
  const int64_t add_bytes = opt.n * 12;
  if (!print_header("02_elementwise / add", opt, add_bytes)) {
    return 2;
  }
  const double peak = peak_gbps();

  DeviceBuffer x(opt.n);
  DeviceBuffer y(opt.n);
  DeviceBuffer out(opt.n);
  fill_pattern(x, 1);
  fill_pattern(y, 2);

  for (const auto& v : kAddVariants) {
    bench_row(v.label, add_bytes, [&] { v.fn(x.get(), y.get(), out.get(), opt.n, 0); }, opt, peak);
  }

  // ------ relu / sigmoid：读 4N + 写 4N = 8 字节/元素 ------
  bench_unary_table("02_elementwise / relu", kReluVariants, 5, x.get(), out.get(), opt, peak);
  bench_unary_table("02_elementwise / sigmoid", kSigmoidVariants, 5, x.get(), out.get(), opt, peak);

  std::printf("\n提示：把本可执行文件直接喂给 ncu 就能拿到硬件计数器，\n");
  std::printf("      例如 ncu --set full ./main_02_elementwise %lld 10\n",
              static_cast<long long>(opt.n));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  return ops_lab::native::run_guarded(run, argc, argv);
}
