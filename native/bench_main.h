// native/bench_main.h
//
// native 线（纯 CUDA 可执行文件）的公共骨架。
//
// ## 这条线存在的理由
//
//   1. `ncu ./main_02_elementwise` 直接出报告 —— 不用穿过 Python 和 torch。性能
//      分析工具对"裸可执行文件"最友好。
//   2. Python 环境 / torch / 扩展任何一环坏了，这条线依然能跑，用来确认
//      "kernel 本身没问题"。
//   3. 它是「kernels/ 零框架依赖」这条铁律的**活证明** —— 哪天有人往 kernel 里
//      塞了 torch 符号，这条线会第一个编译失败。
//
// ## 计时和峰值带宽用的是同一份实现
//
// 全部来自 `kernels/common/`：计时用 `benchmark.h`，设备与峰值带宽用 `device.h`。
// 和 Python 基准（`bench/run_bench.py`）是同一套公式，所以两边数字应当互相印证 ——
// 如果 native 报 240 GB/s 而 Python 报 239.8，说明两条线都可信。
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "kernels/common/benchmark.h"
#include "kernels/common/cuda_check.h"
#include "kernels/common/device.h"

namespace ops_lab {
namespace native {

// ------------------------------------------------------------------ 显存

// 极简 RAII 显存缓冲。native 线不引入 torch，所以分配显存要自己来。
//
// 做成模板是因为 `hello` 的输出是 int32（它算的是索引），不是 float ——
// 一开始写成只支持 float，编译时才暴露出来。
template <typename T>
class DeviceBufferT {
 public:
  explicit DeviceBufferT(int64_t count) : count_(count) {
    if (count > 0) {
      void* raw = nullptr;
      OPSLAB_CUDA_CHECK(cudaMalloc(&raw, static_cast<size_t>(count) * sizeof(T)));
      ptr_ = static_cast<T*>(raw);
    }
  }
  ~DeviceBufferT() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);  // 析构里不抛异常，静默释放
    }
  }
  DeviceBufferT(const DeviceBufferT&) = delete;
  DeviceBufferT& operator=(const DeviceBufferT&) = delete;

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  int64_t count() const { return count_; }

 private:
  T* ptr_ = nullptr;
  int64_t count_ = 0;
};

using DeviceBuffer = DeviceBufferT<float>;
using DeviceBufferI32 = DeviceBufferT<int32_t>;

// 用固定种子的线性同余在 host 侧造数据，再拷上显存。
//
// 为什么不写 device 端 RNG kernel：那是另一个话题，会喧宾夺主 ——
// 这条线要看的是算子的访存性能，不是随机数生成。而且固定种子保证每次比较
// 的都是同一组数据。
inline void fill_pattern(DeviceBuffer& buf, uint32_t seed) {  if (buf.count() <= 0) {
    return;
  }
  std::vector<float> host(static_cast<size_t>(buf.count()));
  uint32_t state = seed;
  for (auto& v : host) {
    state = state * 1664525u + 1013904223u;
    // 取高 24 位映射到 [-1, 1) —— 避免全零/全同值，也避开非规格化数
    v = static_cast<float>(state >> 8) / 8388608.0f - 1.0f;
  }
  OPSLAB_CUDA_CHECK(cudaMemcpy(buf.get(), host.data(), host.size() * sizeof(float),
                               cudaMemcpyHostToDevice));
}

// ------------------------------------------------------------------ 输出

struct NativeOptions {
  int64_t n = 1LL << 24;   // 与 Python 基准的 bench_shapes 对齐
  int warmup = 20;
  int iters = 100;
};

// 解析可选的命令行参数：`<可执行文件> [N] [iters]`
inline NativeOptions parse_options(int argc, char** argv) {
  NativeOptions opt;
  if (argc > 1) {
    opt.n = std::atoll(argv[1]);
  }
  if (argc > 2) {
    opt.iters = std::atoi(argv[2]);
  }
  return opt;
}

// 拿设备信息并打印表头。没有 GPU 时返回 false，调用方应当直接退出。
inline bool print_header(const char* title, const NativeOptions& opt, int64_t bytes) {
  if (device_count() == 0) {
    std::fprintf(stderr, "看不到 CUDA 设备，无法运行。\n");
    std::fprintf(stderr, "（这条线测的就是真实耗时，没有 GPU 就没有替代方案；"
                         "离线检查请用 python tests/run_all.py）\n");
    return false;
  }

  const DeviceInfo info = query_device();
  const double peak = info.mem_bandwidth_gbps;

  std::printf("\n=== %s ===\n", title);
  std::printf("设备        %s (sm_%d%d, %d SM)\n", info.name.c_str(), info.cc_major,
              info.cc_minor, info.sm_count);
  std::printf("N           %lld\n", static_cast<long long>(opt.n));
  std::printf("搬运        %.1f MB\n", static_cast<double>(bytes) / 1e6);
  std::printf("理论峰值    %.1f GB/s\n", peak);
  if (peak > 0) {
    std::printf("理论最短    %.1f us\n", static_cast<double>(bytes) / (peak * 1e9) * 1e6);
  }
  std::printf("计时        warmup %d + %d 次独立 median\n\n", opt.warmup, opt.iters);

  // 表头用 ASCII：printf 按字节计宽，中文表头会让列错位
  std::printf("%-22s %12s %10s %8s\n", "variant", "median(us)", "GB/s", "%peak");
  std::printf("------------------------------------------------------\n");
  return true;
}

// 跑一个变体并打一行。launch 是"只负责启动 kernel"的可调用对象。
template <typename LaunchFn>
void bench_row(const char* label, int64_t bytes, LaunchFn&& launch, const NativeOptions& opt,
               double peak_gbps) {
  const BenchStats stats = bench_launch(launch, opt.warmup, opt.iters);
  const double gbps = achieved_bandwidth_gbps(bytes, stats.median_ms);
  const double pct = percent_of_peak(gbps, peak_gbps);
  std::printf("%-22s %12.1f %10.1f %7.1f%%\n", label, stats.median_ms * 1e3, gbps, pct);
  std::fflush(stdout);
}

inline double peak_gbps() {
  const DeviceInfo info = query_device();
  return info.mem_bandwidth_gbps;
}

}  // namespace native
}  // namespace ops_lab
