// kernels/03_reduction/reduction_ops.cuh
//
// 第 3 章 reduction 的**结合运算**定义。
//
// 一个归约 = 怎么读 + 怎么合并 + 结果给谁。本章五个变体只在"怎么读"
// （v0）和"用多少 block"（v4）以及块内合并方式（v1/v2/v3）上做文章，
// 所以把"合并"这一维抽到这里，换算子时不需要碰任何变体文件。
//
// ## 为什么单位元和 combine 必须放在一起
//
// 归约最容易写错的地方是**空线程**：n 不是 block 大小的整数倍时，有一部分线程
// 一个元素都没分到。它们必须贡献**单位元** —— 贡献 0 或未初始化值都会算错，
// 而且错得非常安静（大多数形状下看不出来）。
//
// 所以单位元不是一个"顺手写的初值"，而是运算定义的一部分。把它和 combine
// 写在同一个结构体里，换运算就同时换掉两者，不可能只改一半。
#pragma once

#include <cuda_runtime.h>

#include <limits>

namespace ops_lab {
namespace reduction {

// ---------------------------------------------------------------------------
// 浮点 atomicMax 不存在，只能自己拼
//
// CUDA 只提供 atomicMax(int*)。浮点版本要用 CAS：读出旧值 → 算出新值 →
// atomicCAS 尝试交换 → 失败就重来。
//
// 关键细节：比较一律在**浮点域**做（fmaxf）。float 的 IEEE 位模式在非负区间
// 与整数序一致，一遇到负数就反了（-0.0f 的位模式是 0x80000000，比 +0.0f 的
// 0x00000000 大）。所以位模式只当"要交换的载荷"，不当比较依据。
//
// 本章只要求它能正确工作：CAS 循环只发生在"每个 block 一次"的频率上
// （本机 144 个 block → 144 次），不会成为瓶颈。
// ---------------------------------------------------------------------------
__device__ __forceinline__ void atomic_max_float(float* addr, float value) {
  int* addr_as_int = reinterpret_cast<int*>(addr);
  int old = *addr_as_int;
  int assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(addr_as_int, assumed, __float_as_int(fmaxf(value, __int_as_float(assumed))));
  } while (assumed != old);
}

// ---------------------------------------------------------------------- sum
struct SumOp {
  // 单位元：加法的 0。n == 0 时它就是最终输出（空数组的和是 0）。
  static constexpr float kIdentity = 0.0f;

  static __device__ __forceinline__ float combine(float a, float b) { return a + b; }

  static __device__ __forceinline__ void atomic_combine(float* out, float value) {
    atomicAdd(out, value);
  }
};

// ---------------------------------------------------------------------- max
struct MaxOp {
  // 单位元：负无穷。**不能写 0.0f** —— 全负数输入时 max 会错误地给出 0。
  // 本章的测试输入生成器专门造带负数的数据，就是为了让这个错误暴露出来。
  //
  // 这里用 std::numeric_limits 而不是 CUDART_INF_F：后者是
  // `__int_as_float(0x7f800000U)`，是个**设备端函数**，没法在 host 侧的
  // 常量表达式里用；而 launcher 需要把这个单位元作为参数传给 init kernel，
  // 所以它必须是 host 和 device 都能取的值。
  static constexpr float kIdentity = -std::numeric_limits<float>::infinity();

  static __device__ __forceinline__ float combine(float a, float b) { return fmaxf(a, b); }

  static __device__ __forceinline__ void atomic_combine(float* out, float value) {
    atomic_max_float(out, value);
  }
};

}  // namespace reduction
}  // namespace ops_lab
