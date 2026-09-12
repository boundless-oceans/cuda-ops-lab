// kernels/common/benchmark.h
//
// CUDA event 计时。native/ 的纯 CUDA 可执行文件用这个，
// bench/run_bench.py 走的是 binding 里的同名封装（同一套逻辑，避免两处漂移）。
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>

#include "cuda_check.h"

namespace ops_lab {

struct BenchStats {
  int iters = 0;
  double min_ms = 0.0;
  double median_ms = 0.0;
  double mean_ms = 0.0;
  double max_ms = 0.0;
};

inline double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double idx = p * static_cast<double>(v.size() - 1);
  const size_t lo = static_cast<size_t>(idx);
  const size_t hi = std::min(lo + 1, v.size() - 1);
  const double frac = idx - static_cast<double>(lo);
  return v[lo] * (1.0 - frac) + v[hi] * frac;
}

// launch 是一个可调用对象，内部只负责「启动 kernel」。
//
// 为什么每次迭代单独 record/synchronize 而不是「记录一次、循环 N 次、再记录」？
//   因为我们要的是 median，不是平均值。单次同步能拿到每次迭代的独立样本，
//   这样一次偶发的调度抖动只会污染一个样本，不会污染整组数据。
//   代价是每次迭代多了 ~5µs 的同步开销 —— 对我们的算子（数十微秒起）可以忽略。
//   对 <10µs 的小 kernel，请直接用 run_bench.py（torch 侧也一样会有这个开销）。
template <typename LaunchFn>
BenchStats bench_launch(LaunchFn&& launch, int warmup = 20, int iters = 100,
                        cudaStream_t stream = nullptr) {
  for (int i = 0; i < warmup; ++i) launch();
  OPSLAB_CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  OPSLAB_CUDA_CHECK(cudaEventCreate(&start));
  OPSLAB_CUDA_CHECK(cudaEventCreate(&stop));

  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    OPSLAB_CUDA_CHECK(cudaEventRecord(start, stream));
    launch();
    OPSLAB_CUDA_CHECK(cudaEventRecord(stop, stream));
    OPSLAB_CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    OPSLAB_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    samples.push_back(static_cast<double>(ms));
  }

  OPSLAB_CUDA_CHECK(cudaEventDestroy(start));
  OPSLAB_CUDA_CHECK(cudaEventDestroy(stop));

  BenchStats stats;
  stats.iters = iters;
  if (!samples.empty()) {
    stats.min_ms = *std::min_element(samples.begin(), samples.end());
    stats.max_ms = *std::max_element(samples.begin(), samples.end());
    stats.median_ms = percentile(samples, 0.5);
    double sum = 0.0;
    for (double s : samples) sum += s;
    stats.mean_ms = sum / static_cast<double>(samples.size());
  }
  return stats;
}

// 把耗时换算成有效带宽（GB/s）。bytes 是这次 kernel 实际搬运的总字节数
// （读 + 写，见每章 OPS 元数据里的 bytes 定义）。
inline double achieved_bandwidth_gbps(int64_t bytes, double ms) {
  if (ms <= 0.0) return 0.0;
  return static_cast<double>(bytes) / (ms * 1e-3) / 1e9;
}

// 离理论峰值还有多远
inline double percent_of_peak(double achieved_gbps, double peak_gbps) {
  if (peak_gbps <= 0.0) return 0.0;
  return 100.0 * achieved_gbps / peak_gbps;
}

}  // namespace ops_lab
