// kernels/02_elementwise/elementwise_ops.cuh
//
// elementwise 算子的**数学本体**：每个算子就一个 device 函数。
//
// 为什么要单独一个文件：add / relu / sigmoid 三个算子共用同一套 kernel 模板
// （见各元素变体文件），变体文件里的 kernel 只负责"怎么访存"，
// 这里只负责"算什么"。这样"3 个算子"不等于 3 倍代码量。
#pragma once

#include <cuda_runtime.h>

namespace ops_lab {
namespace elementwise {

// --- 二目算子：out = f(a, b) ---
struct AddOp {
  static __device__ __forceinline__ float apply(float a, float b) { return a + b; }
};

// --- 一目算子：out = f(a) ---
struct ReluOp {
  static __device__ __forceinline__ float apply(float a) { return a > 0.0f ? a : 0.0f; }
};

struct SigmoidOp {
  // 用 expf 而不是 __expf（快速近似）。
  //
  // __expf 的误差随 |x| 增长（CUDA 文档给的界是 2 + floor(|1.16x|) ulp），
  // 在 x = -10 附近就已经十几个 ulp，拿它和 float64 参考实现比对时，
  // 容差会变成"设多少都心虚"。
  //
  // 代价是 exp 从一条 SFU 指令变成一小段软件序列 —— 这正好让 sigmoid
  // 成为本章唯一"算术强度明显上升"的算子，可以用来验证：
  // 算术强度上升了，它是不是**仍然** memory-bound？
  // 若实测远慢于按带宽算出的理论值，说明瓶颈已经不在带宽上。
  static __device__ __forceinline__ float apply(float a) {
    return 1.0f / (1.0f + expf(-a));
  }
};

}  // namespace elementwise
}  // namespace ops_lab
