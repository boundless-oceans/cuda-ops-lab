// kernels/01_execution/execution.cuh
//
// 第 1 章：执行模型。
//
// 这一章不是"优化阶梯"，而是**两个受控实验** —— 它不追求带宽数字，
// 只负责建立后面 10 章都要用的心智模型。
//
//   实验一 hello            grid/block/thread 的映射，以及"grid 有没有上限"
//   实验二 bandwidth_probe  occupancy 受控实验：**kernel 代码完全相同**，
//                           只改动态共享内存占用，看带宽怎么变
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "../common/device.h"

namespace ops_lab {
namespace execution {

constexpr int kBlockSize = 256;

// ---------------------------------------------------------------------------
// 实验一：hello
// ---------------------------------------------------------------------------

// v1_naive —— 一线程一元素，精确 grid，带边界检查。
//   限制：没有 grid-stride 循环，所需 block 数超过 gridDim.x 上限（2^31-1）
//         时会抛清晰错误。这是**故意保留**的，因为 v2 的存在意义就是解除它。
//
//   注意 out 是 int32：本函数是索引演示，n 超过 int32 范围时结果无意义。
void hello_v1_naive(int32_t* out, int64_t n, cudaStream_t stream);

// v2_grid_stride —— grid 固定，kernel 内循环步进。
//   与 v1 的差别**不是快慢，而是"grid 有没有上限"**。
//   这也正是第 2 章 v3_grid_stride 存在的理由。
//
//   grid_size <= 0 表示"按设备规模自动选"（`default_grid_size`）。
//   **这个约定由 launcher 自己实现**，调用方原样传值即可 —— 曾经它只在 Python
//   绑定层实现，导致 native 线传 0 时直接抛异常。
void hello_v2_grid_stride(int32_t* out, int64_t n, int block_size, int grid_size,
                          cudaStream_t stream);

// ---------------------------------------------------------------------------
// 实验二：bandwidth_probe —— occupancy 受控实验
// ---------------------------------------------------------------------------
//
// 四个变体是**同一份拷贝逻辑**，唯一区别是动态共享内存占用：
//
//   变体 1  v1_smem0    0 KB      不声明 extern __shared__
//   变体 2  v2_smem24k  24 KB
//   变体 3  v3_smem32k  32 KB
//   变体 4  v4_smem48k  48 KB
//
// 为什么变体 1 要用另一份 kernel：如果同一份代码声明了 extern __shared__
// 却以 0 字节启动、还去访问 smem[0]，那是未定义行为。
//
// 预期（先写下来，等实测打脸或确认）：
//   occupancy 从 ~100% 压到 ~33%，**带宽基本不变** —— 纯流式拷贝在
//   1/3 的占用率下仍能打满带宽（Little's law：每线程多个在飞访存请求
//   可以补偿线程数不足）。若实测明显下降，说明 grid 太小 / 每线程并行度
//   不够，而不是"occupancy 不够"。

constexpr int kProbeVariantCount = 4;

void bandwidth_probe(const float* x, float* out, int64_t n, int variant, cudaStream_t stream);

// 该变体使用的动态共享内存字节数
int bandwidth_probe_smem_bytes(int variant);

// 该变体在给定 block size 下的理论 occupancy（host 侧 API，无需启动 kernel）
Occupancy bandwidth_probe_occupancy(int variant, int block_size);

}  // namespace execution
}  // namespace ops_lab
