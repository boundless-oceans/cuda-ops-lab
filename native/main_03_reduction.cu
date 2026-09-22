// native/main_03_reduction.cu
//
// 第 3 章的可执行文件：不经过 Python / torch，直接跑 reduction 五级阶梯。
//
//   ./main_03_reduction                    # 默认 N = 2^24
//   ./main_03_reduction 1048576 200        # 指定 N 与迭代次数
//   ncu -k "regex:reduce_v1_naive" --launch-count 1 --set full ./main_03_reduction
//
// 数字应当与 `python bench/run_bench.py --chapters reduction` 互相印证 ——
// 两边用的是同一份 kernels/ 和同一套计时/峰值公式。
//
// ## 这条线比 Python 线多做的一件事：结果自检
//
// 它会把每个变体的输出和 **host 侧的 double 参考值**比一遍。理由是实用主义的：
// 这个可执行文件是拿去喂 ncu 的，如果某个变体悄悄算错了，我们会对着一堆
// 错误结果做性能分析，还以为是"优化空间很大"。数据本来就是 host 造的
// （`make_host_pattern`），参考值顺手就能算，不需要引入任何东西。
//
// 参考值本身也是"两条独立路径"的一部分：Python 线用 torch 的 float64 归约，
// 这条线用 host 上的手写 double 循环。两条都对上了，才敢说 kernel 是对的。
#include <cstdio>
#include <limits>
#include <vector>

#include "native/bench_main.h"

#include "kernels/03_reduction/reduction.cuh"

namespace {

using ReduceFn = void (*)(const float*, float*, int64_t, cudaStream_t);

struct Variant {
  const char* label;
  ReduceFn fn;
};

constexpr int kNumVariants = 5;

const Variant kSumVariants[kNumVariants] = {
    {"v0_uncoalesced", ops_lab::reduction::sum_v0_uncoalesced},
    {"v1_naive", ops_lab::reduction::sum_v1_naive},
    {"v2_smem_tree", ops_lab::reduction::sum_v2_smem_tree},
    {"v3_warp_shuffle", ops_lab::reduction::sum_v3_warp_shuffle},
    {"v4_single_block", ops_lab::reduction::sum_v4_single_block},
};

const Variant kMaxVariants[kNumVariants] = {
    {"v0_uncoalesced", ops_lab::reduction::max_v0_uncoalesced},
    {"v1_naive", ops_lab::reduction::max_v1_naive},
    {"v2_smem_tree", ops_lab::reduction::max_v2_smem_tree},
    {"v3_warp_shuffle", ops_lab::reduction::max_v3_warp_shuffle},
    {"v4_single_block", ops_lab::reduction::max_v4_single_block},
};

// ------------------------------------------------------------------- 自检

struct Reference {
  double sum = 0.0;
  double max = 0.0;
  double abs_sum = 0.0;  // Σ|x_i|，用来给 sum 定一个有理论依据的判据尺度
};

Reference host_reference(const std::vector<float>& host) {
  Reference r;
  r.max = -std::numeric_limits<double>::infinity();
  for (const float v : host) {
    const double d = static_cast<double>(v);
    r.sum += d;
    r.abs_sum += (d < 0.0 ? -d : d);
    r.max = (d > r.max) ? d : r.max;
  }
  return r;
}

// 跑一遍、把结果读回来、与参考值比对，返回是否通过。
bool check_variant(const char* label, ReduceFn fn, const float* x, float* out, int64_t n,
                   double expected, double tolerance) {
  fn(x, out, n, /*stream=*/0);
  float got = 0.0f;
  OPSLAB_CUDA_CHECK(cudaMemcpy(&got, out, sizeof(float), cudaMemcpyDeviceToHost));

  const double got_d = static_cast<double>(got);
  const double diff = got_d > expected ? got_d - expected : expected - got_d;
  const bool ok = diff <= tolerance;
  std::printf("  %-18s 实际 %-16.8g 参考 %-16.8g %s\n", label, got_d, expected,
              ok ? "OK" : "<<< 不一致");
  return ok;
}

void bench_table(const char* title, const Variant* variants, const float* x, float* out, int64_t n,
                 const ops_lab::native::NativeOptions& opt, double peak) {
  const int64_t bytes = n * 4 + 4;  // 读 4N + 写 4（输出只有一个数）
  if (!ops_lab::native::print_header(title, opt, bytes)) {
    return;
  }
  for (int i = 0; i < kNumVariants; ++i) {
    ops_lab::native::bench_row(
        variants[i].label, bytes, [&] { variants[i].fn(x, out, n, 0); }, opt, peak);
  }
}

}  // namespace

namespace {

int run(int argc, char** argv) {
  using namespace ops_lab::native;

  const NativeOptions opt = parse_options(argc, argv);

  // 必须在分配显存**之前**问这一句：DeviceBuffer 的构造函数就会 cudaMalloc
  if (require_device()) {
    return 2;
  }

  // 造数据：host 侧生成 → 算参考值 → 拷上显存。三处用的是同一组值。
  const std::vector<float> host = make_host_pattern(opt.n, /*seed=*/20260914);
  const Reference ref = host_reference(host);

  DeviceBuffer x(opt.n);
  DeviceBuffer out(1);
  upload_pattern(x, host);

  std::printf("\n=== 03_reduction / 结果自检（不依赖 Python）===\n");
  std::printf("host 参考值（double）  sum = %.8g   max = %.8g   Σ|x| = %.8g\n", ref.sum, ref.max,
              ref.abs_sum);
  // sum 的判据取"浮点累加误差上界"的量级，而不是相对于 sum 的比值：
  // 本函数的填充模式在 [-1,1) 上均匀分布，求和的期望是 0，用比值判据会退化成一个
  // 没有意义的数。用 Σ|x| 当尺度，对任何输入都成立。
  //
  // 这里刻意留了较大余量（1e-4·Σ|x|）：自检的目的是抓**粗错**（少算一整块、
  // 根本没启动、单位元写错、索引公式错到分区不覆盖），不是做精度测试 ——
  // 精度和"恰好覆盖一次"由 `tests/test_03_reduction_edge.py` 的全 1 输入负责，
  // 那种输入下 float32 求和是精确的，可以要求逐位相等。
  const double sum_tolerance = 1e-4 * ref.abs_sum + 1e-6;
  std::printf("判据  sum: |误差| <= %.4g（1e-4·Σ|x|）；max: 精确相等（只有比较，没有算术）\n\n",
              sum_tolerance);

  bool all_ok = true;
  for (const Variant& v : kSumVariants) {
    if (!check_variant(v.label, v.fn, x.get(), out.get(), opt.n, ref.sum, sum_tolerance)) {
      all_ok = false;
    }
  }
  for (const Variant& v : kMaxVariants) {
    if (!check_variant(v.label, v.fn, x.get(), out.get(), opt.n, ref.max, 0.0)) {
      all_ok = false;
    }
  }

  if (!all_ok) {
    std::fprintf(stderr, "\n[警告] 有变体的结果与 host 参考值不一致。\n");
    std::fprintf(stderr, "        计时仍会打印，但**不要**拿这些数字做性能分析 ——\n");
    std::fprintf(stderr, "        算错的结果也可能来自被编译器优化掉或提前退出的代码。\n");
  }

  const double peak = peak_gbps();

  // 计时之前把 GPU 推到时钟稳态：冷态显存时钟 7001 MHz、稳态 8001 MHz，差 14%，
  // 而理论峰值是按额定的 8001 算的。没有这一步，先测的变体会被低估约 14%
  // （第 3 章的两条测量路径就是这么差出 14% 来的）。
  // 用 v1_naive 当预热负载：它就是"打满带宽"的那一档，与真实测量同型。
  steady_state_warmup([&] { ops_lab::reduction::sum_v1_naive(x.get(), out.get(), opt.n, 0); },
                      2.0);

  bench_table("03_reduction / sum", kSumVariants, x.get(), out.get(), opt.n, opt, peak);
  bench_table("03_reduction / max", kMaxVariants, x.get(), out.get(), opt.n, opt, peak);

  std::printf("\n提示：把本可执行文件直接喂给 ncu 就能拿到硬件计数器，\n");
  std::printf("      例如 ncu -k \"regex:reduce_v0\" --launch-count 1 --set full "
              "./main_03_reduction %lld 10\n",
              static_cast<long long>(opt.n));
  return all_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) { return ops_lab::native::run_guarded(run, argc, argv); }
