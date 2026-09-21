// kernels/03_reduction/reduction_config.cuh
//
// 第 3 章的启动配置常量。**这个文件里只有常量，没有任何 device 代码。**
//
// 为什么要把常量单独拆出来：`bindings/bind_03_reduction.cpp` 是**宿主**编译单元，
// 它要 include `reduction.cuh` 才能拿到 launcher 声明。如果 `reduction.cuh` 顺手
// 把带 `__device__` 函数的头也 include 进来，宿主编译器就会在
// `threadIdx` / `__syncthreads` / `atomicAdd` 上报一堆"未声明" ——
// 实测就是这么炸的。
//
// 所以本章的头文件分成两组，别搞混：
//
//   reduction_config.cuh   常量            ← 谁都能 include
//   reduction.cuh          launcher 声明   ← 只有宿主侧和 .cu 能 include
//   reduction_ops.cuh      运算 + 浮点 atomicMax  ← 只有 .cu 能 include
//   reduction_block.cuh    块内归约三种写法       ← 只有 .cu 能 include
//
// 第 2 章之所以没这个问题，是因为它的 `elementwise.cuh` 里只有声明，
// 设备代码全在各自的 .cu 里。
#pragma once

namespace ops_lab {
namespace reduction {

// 块大小。256 = 8 个 warp：块内树 8 级；warp shuffle 版第一级 5 次 shuffle、
// 第二级只剩 3 次，同步次数从 8 次降到 1 次。
constexpr int kBlockSize = 256;
constexpr int kNumWarps = kBlockSize / 32;

static_assert(kBlockSize % 32 == 0, "kBlockSize 必须是 32 的整数倍");
static_assert(kNumWarps >= 2, "warp 数少于 2 时第二级 shuffle 要另写特例，本仓库不做");

}  // namespace reduction
}  // namespace ops_lab
