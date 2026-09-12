// kernels/common/device.h
//
// 设备信息查询 + 理论峰值推算。
//
// 这个文件的存在意义是让「理论下限」不是拍脑袋：每章 README 第 2 段要写
// "N = 2^20 时理论最小耗时 ≈ 16.4 µs"，这个 16.4 就该从这里算出来，
// 而不是从别人的博客里抄。GPU 换了，数字自动变。
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "cuda_check.h"

namespace ops_lab {

struct DeviceInfo {
  int ordinal = 0;
  std::string name;
  int cc_major = 0;
  int cc_minor = 0;
  int sm_count = 0;
  int max_threads_per_sm = 0;
  int max_threads_per_block = 0;
  int warp_size = 32;
  int regs_per_sm = 0;
  int regs_per_block = 0;
  int smem_per_sm = 0;        // 每 SM 可用共享内存（字节）
  int smem_per_block_optin = 0;  // cudaFuncAttributeMaxDynamicSharedMemorySize 上限
  int mem_clock_khz = 0;
  int mem_bus_width = 0;      // bit
  double mem_bandwidth_gbps = 0;  // 理论峰值带宽
};

// 理论峰值带宽 = 显存频率 × 位宽 / 8 × 2（DDR 双边沿）
// 例：RTX 4060 Laptop = 8 Gbps × 128 bit / 8 = 128 GB/s ... 实际颗粒等效
// 8000 MT/s × 128 / 8 = 128 GB/s；手册写的 ~256 GB/s 是 GPU Boost 下的等效值。
// 这里用 memClockRate（kHz）× busWidth 算出来的值偏保守，对"判断离上限多远"
// 反而更安全 —— 我们宁可低估峰值，也不要高估自己。
inline double compute_bandwidth_gbps(int mem_clock_khz, int bus_width_bits) {
  // kHz * bit / 8 -> bytes/s，再乘 2 表示 DDR 双边沿，最后转 GB/s
  const double bytes_per_sec =
      static_cast<double>(mem_clock_khz) * 1000.0 * (bus_width_bits / 8.0) * 2.0;
  return bytes_per_sec / 1e9;
}

inline DeviceInfo query_device(int ordinal = 0) {
  DeviceInfo info;
  info.ordinal = ordinal;

  cudaDeviceProp prop{};
  OPSLAB_CUDA_CHECK(cudaGetDeviceProperties(&prop, ordinal));

  info.name = prop.name;
  info.cc_major = prop.major;
  info.cc_minor = prop.minor;
  info.sm_count = prop.multiProcessorCount;
  info.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
  info.max_threads_per_block = prop.maxThreadsPerBlock;
  info.warp_size = prop.warpSize;
  info.regs_per_sm = prop.regsPerMultiprocessor;
  info.regs_per_block = prop.regsPerBlock;
  info.smem_per_sm = static_cast<int>(prop.sharedMemPerMultiprocessor);
  info.smem_per_block_optin = static_cast<int>(prop.sharedMemPerBlockOptin);
  info.mem_clock_khz = prop.memoryClockRate;
  info.mem_bus_width = prop.memoryBusWidth;
  info.mem_bandwidth_gbps = compute_bandwidth_gbps(prop.memoryClockRate, prop.memoryBusWidth);

  return info;
}

// 当前进程可见的 GPU 数量。沙箱/无驱动环境返回 0 而不是抛异常，
// 这样 check_env.py 能优雅地报告"无 GPU"，而不是崩溃。
inline int device_count() {
  int n = 0;
  const cudaError_t err = cudaGetDeviceCount(&n);
  if (err != cudaSuccess) {
    cudaGetLastError();  // 清掉 sticky error
    return 0;
  }
  return n;
}

// 某 kernel 在给定 block size / 动态 smem 下的理论 occupancy。
// 这是第 1 章的核心实验：不改一行 kernel 代码，只改 smem 用量，
// 就能看到 occupancy 从 100% 掉到 25%，然后观察带宽怎么跟着掉。
struct Occupancy {
  int block_size = 0;
  int dynamic_smem_bytes = 0;
  int blocks_per_sm = 0;             // 每 SM 能同时驻留的 block 数
  int active_warps_per_sm = 0;
  int max_warps_per_sm = 0;
  double occupancy = 0.0;            // 0~1
};

inline Occupancy query_occupancy(const void* kernel, int block_size, int dynamic_smem_bytes) {
  Occupancy oc;
  oc.block_size = block_size;
  oc.dynamic_smem_bytes = dynamic_smem_bytes;

  int blocks = 0;
  OPSLAB_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, kernel, block_size, static_cast<size_t>(dynamic_smem_bytes)));
  oc.blocks_per_sm = blocks;

  cudaDeviceProp prop{};
  int dev = 0;
  OPSLAB_CUDA_CHECK(cudaGetDevice(&dev));
  OPSLAB_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

  oc.max_warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;
  oc.active_warps_per_sm = blocks * (block_size / prop.warpSize);
  oc.occupancy = oc.max_warps_per_sm > 0
                     ? static_cast<double>(oc.active_warps_per_sm) / oc.max_warps_per_sm
                     : 0.0;
  return oc;
}

}  // namespace ops_lab
