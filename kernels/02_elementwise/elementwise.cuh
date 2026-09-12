// kernels/02_elementwise/elementwise.cuh
//
// 第 2 章 elementwise 的 host launcher 声明。
//
// 铁律（design §5.1）：这个头文件以及所有 .cu 里**不出现任何 torch 符号**。
// kernel 只认裸指针和长度，框架适配是 bindings/ 那一层的事。
// 这不只是洁癖 —— Ascend C 的 kernel 同样只认 GM 指针，这层分离就是对比赛的结构预演。
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ops_lab {
namespace elementwise {

// 块大小集中在此，做实验时只改一处。
constexpr int kBlockSize = 256;

// ---------------------------------------------------------------------------
// v1_naive —— 正确但朴素，作为整条阶梯的基准点
//
// 思路
//   全局线程 id 直接映射到一个元素；相邻线程访问相邻地址，因此访存天然合并；
//   带边界检查处理尾部。
//
// 限制（故意保留）
//   没有 grid-stride 循环。若所需 block 数超过 gridDim.x 上限（2^31-1），
//   会抛出**清晰的错误**而不是静默出错 —— 让"grid 有上限"这件事在测试里可见。
//   解除该限制是 v3_grid_stride 的存在意义。
//
// 预期
//   合并访存已达成，但每个线程只有一个在飞的访存请求，并发度不足
//   → 约 60~75% 峰值带宽（这是阶梯的第二级，不是终点）。
//
// 对应 Ascend C
//   Ascend C 的做法是"先用 DataCopy 把整块搬进 UB，再用向量指令批量算"。
//   本函数"一线程一元素"的结构**不能**直接平移 —— 这正是 design §9 强调的
//   思维转换点。
// ---------------------------------------------------------------------------
void add_v1_naive(const float* x, const float* y, float* out, int64_t n, cudaStream_t stream);

}  // namespace elementwise
}  // namespace ops_lab
