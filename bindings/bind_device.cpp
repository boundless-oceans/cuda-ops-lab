// bindings/bind_device.cpp
//
// 设备信息的基础设施绑定。
//
// 为什么需要它：算 %峰值带宽必须有**显存位宽**，而 nvidia-smi 在新驱动上
// 已经不输出 Bus Width 了（实测 driver 580.178.04 / CUDA 13.0 的
// `nvidia-smi -q -d MEMORY` 只有 FB / BAR1 / Conf Compute 三段）。
// cudaDeviceProp 里有 memoryBusWidth，`kernels/common/device.h` 早就在读它，
// 只是没暴露给 Python。
//
// 顺带的好处：这里的值就是权威值 —— `native/` 的纯 CUDA 可执行文件用的是
// 同一份 device.h，两边不会给出不同的"理论峰值"。
#include <pybind11/pybind11.h>

#include "bindings/bind_helpers.h"
#include "kernels/common/device.h"

namespace py = pybind11;

namespace {

py::dict device_info() {
  const ops_lab::DeviceInfo info = ops_lab::query_device();

  py::dict d;
  d["name"] = info.name;
  d["cc_major"] = info.cc_major;
  d["cc_minor"] = info.cc_minor;
  d["sm_count"] = info.sm_count;
  d["max_threads_per_sm"] = info.max_threads_per_sm;
  d["max_threads_per_block"] = info.max_threads_per_block;
  d["warp_size"] = info.warp_size;
  d["regs_per_sm"] = info.regs_per_sm;
  d["regs_per_block"] = info.regs_per_block;
  d["smem_per_sm"] = info.smem_per_sm;
  d["smem_per_block_optin"] = info.smem_per_block_optin;
  d["mem_clock_khz"] = info.mem_clock_khz;
  d["mem_bus_width"] = info.mem_bus_width;
  d["mem_bandwidth_gbps"] = info.mem_bandwidth_gbps;
  return d;
}

}  // namespace

namespace ops_lab {

void bind_device(py::module_& m) {
  bind_kernel(m, "device_info", "00_infrastructure", "device_info",
              "设备信息：SM 数、寄存器、共享内存、显存时钟与位宽、理论峰值带宽",
              &device_info);
}

}  // namespace ops_lab
