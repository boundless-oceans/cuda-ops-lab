// native/main_04_scan.cu
//
// 第 4 章的可执行文件：不经过 Python / torch，直接跑 scan 六级阶梯。
//
//   ./main_04_scan                      # 默认 N = 2^23
//   ./main_04_scan 1048576 200          # 指定 N 与迭代次数
//   ncu -k "regex:scan_single_pass_kernel.*<8>" --launch-count 1 --set full ./main_04_scan
//
// ## 这条线做两件 Python 线之外的事
//
// ① **独立的结果自检**：用**全 1 输入**。它的 inclusive scan 恰好是 1,2,3,…，
//    而 n < 2^24 时 float32 能精确表示这些整数 —— 所以判据可以是**逐位相等**。
//    这对 scan 特别重要：任何一个 chunk 的前缀算错，都会让它**后面全体偏移**。
//    随机输入配 1e-3 容差时，这类错误会被吃掉一部分；全 1 输入无处可藏。
//    （真事：两级块内扫描重构时，我一度把"线程本地和"当成 chunk 总和发布出去，
//     就是这条自检先炸的。）
//
// ② 顺带交叉印证 Python 线的数字（两边同一份 kernels/、同一套计时公式）。
#include <cstdio>
#include <functional>
#include <vector>

#include "native/bench_main.h"

#include "kernels/04_scan/scan.cuh"

namespace {

using ScanCall = std::function<void(const float*, float*, int64_t, float*)>;

struct Variant {
  const char* label;
  ScanCall call;
  int passes;  // 全局读写遍数：三段式 3 遍（12N），单趟 2 遍（8N）
};

const Variant kVariants[] = {
    {"v0_naive", [](const float* x, float* o, int64_t n, float* ws) {
       ops_lab::scan::scan_inclusive_v0_naive(x, o, n, ws, 0);
     }, 3},
    {"v1_kogge_stone", [](const float* x, float* o, int64_t n, float* ws) {
       ops_lab::scan::scan_inclusive_v1_kogge_stone(x, o, n, ws, 0);
     }, 3},
    {"v2_blelloch", [](const float* x, float* o, int64_t n, float* ws) {
       ops_lab::scan::scan_inclusive_v2_blelloch(x, o, n, ws, 0);
     }, 3},
    {"v3_single_pass", [](const float* x, float* o, int64_t n, float* ws) {
       ops_lab::scan::scan_inclusive_v3_single_pass(x, o, n, ws, 0);
     }, 2},
    {"v4_optimized", [](const float* x, float* o, int64_t n, float* ws) {
       ops_lab::scan::scan_inclusive_v4_optimized(x, o, n, ws, 0);
     }, 2},
    {"v5_single_block", [](const float* x, float* o, int64_t n, float* ws) {
       ops_lab::scan::scan_inclusive_v5_single_block(x, o, n, 0);
       (void)ws;
     }, 2},
};

// ------------------------------------------------------------------ 自检

// 全 1 输入下，inclusive scan 的第 i 个输出必须是精确的 i+1。
// 返回第一个不合格的下标（全都合格返回 -1）。
int64_t check_ones(const float* x, float* out, int64_t n, const ScanCall& call, float* ws) {
  call(x, out, n, ws);
  std::vector<float> host(static_cast<size_t>(n));
  OPSLAB_CUDA_CHECK(cudaMemcpy(host.data(), out, sizeof(float) * host.size(),
                               cudaMemcpyDeviceToHost));
  for (int64_t i = 0; i < n; ++i) {
    const float want = static_cast<float>(i + 1);
    if (host[static_cast<size_t>(i)] != want) {
      return i;
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

  const int64_t ws_floats = ops_lab::scan::workspace_floats(opt.n);
  DeviceBuffer ones_buf(opt.n);
  DeviceBuffer self_out(opt.n);
  DeviceBuffer ws(ws_floats);

  // ---------------------------------------------------------- ① 结果自检
  const std::vector<float> ones(static_cast<size_t>(opt.n), 1.0f);
  upload_pattern(ones_buf, ones);

  std::printf("\n=== 04_scan / 结果自检（全 1 输入，判据 = 逐位相等）===\n");
  std::printf("n = %lld < 2^24，所以 1,2,3,… 在 float32 里可以精确表示。\n",
              static_cast<long long>(opt.n));
  std::printf("任何一段前缀算错，都会让它**后面全体偏移** —— 随机输入的容差会吃掉一部分，\n");
  std::printf("全 1 输入不会。\n\n");

  bool all_ok = true;
  for (const Variant& v : kVariants) {
    const int64_t bad = check_ones(ones_buf.get(), self_out.get(), opt.n, v.call, ws.get());
    if (bad < 0) {
      std::printf("  %-18s OK\n", v.label);
    } else {
      float got = 0.0f;
      OPSLAB_CUDA_CHECK(cudaMemcpy(&got, self_out.get() + bad, sizeof(float),
                                   cudaMemcpyDeviceToHost));
      std::printf("  %-18s 第 %lld 个元素不对：实际 %.1f，期望 %.1f  <<< 不一致\n", v.label,
                  static_cast<long long>(bad), static_cast<double>(got),
                  static_cast<double>(bad + 1));
      all_ok = false;
    }
  }
  if (!all_ok) {
    std::fprintf(stderr, "\n[警告] 有变体的结果与 host 参考不一致。\n");
    std::fprintf(stderr, "        计时仍会打印，但**不要**拿这些数字做性能分析。\n");
  }

  // ------------------------------------------------ ② 稳态预热 + 计时表
  const std::vector<float> data = make_host_pattern(opt.n, /*seed=*/20260914);
  DeviceBuffer x(opt.n);
  DeviceBuffer out(opt.n);
  upload_pattern(x, data);

  steady_state_warmup([&] { ops_lab::scan::scan_inclusive_v4_optimized(
                                x.get(), out.get(), opt.n, ws.get(), 0); },
                      2.0);

  const double peak = peak_gbps();

  // 表头按**理想**流量（8N）打印；每一行用自己的实际搬运量算 GB/s 与 %峰值。
  // 各变体的流量不同（三段式 12N、单趟 8N），用同一个分母就是拿两把尺子量东西
  // —— 见 scan.cuh 的文件头。
  if (!print_header("04_scan / inclusive_scan（理想流量 = 8N，即读一遍写一遍）", opt,
                    4 * 2 * opt.n)) {
    return 2;
  }
  for (const Variant& v : kVariants) {
    const int64_t bytes = 4 * v.passes * opt.n;  // 每元素 4 字节 × 遍数
    bench_row(v.label, bytes, [&] { v.call(x.get(), out.get(), opt.n, ws.get()); }, opt, peak);
  }

  std::printf("\n提示：把本可执行文件直接喂给 ncu 就能拿到硬件计数器，\n");
  std::printf("      最值得看的是 v3 与 v4 —— 它们**只差每线程元素数**：\n");
  std::printf("      ncu -k \"regex:scan_single_pass_kernel\" --launch-count 1 --set full "
              "./main_04_scan %lld 10\n",
              static_cast<long long>(opt.n));
  return all_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) { return ops_lab::native::run_guarded(run, argc, argv); }
