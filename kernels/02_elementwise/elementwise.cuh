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
void relu_v1_naive(const float* x, float* out, int64_t n, cudaStream_t stream);
void sigmoid_v1_naive(const float* x, float* out, int64_t n, cudaStream_t stream);

// ---------------------------------------------------------------------------
// v0_uncoalesced —— 反面教材
//
// 思路
//   索引写成 i = threadIdx.x * gridDim.x + blockIdx.x，相邻线程的地址因此
//   相隔 gridDim.x 个元素。一个 warp 的 32 个线程落在 32 条不同的 cache line
//   上，一次访存请求被拆成 32 个事务。
//
// 与 v1_naive 的关系
//   **只差索引公式那一行**，其余（grid 大小、边界检查、无循环）完全相同，
//   所以阶梯表里 v0 → v1 的差距只能归因于访存模式。
//
// 预期
//   有效带宽 ≈ 峰值的 10~20%。这是本章唯一"故意写错"的一级，存在的意义是
//   让"未合并访存"这件事有一个可测量的代价，而不是一句口号。
// ---------------------------------------------------------------------------
void add_v0_uncoalesced(const float* x, const float* y, float* out, int64_t n,
                        cudaStream_t stream);
void relu_v0_uncoalesced(const float* x, float* out, int64_t n, cudaStream_t stream);
void sigmoid_v0_uncoalesced(const float* x, float* out, int64_t n, cudaStream_t stream);

// ---------------------------------------------------------------------------
// v2_unroll4 —— 提高每线程的指令级并行（ILP）
//
// 思路
//   线程数降到 1/4，但每个线程一口气处理 4 个元素：先把 4 个 x 和 4 个 y
//   全部读进来（8 个互不依赖的 load），再统一计算、统一写回。
//   总并发度 = 线程数 × 每线程在飞请求数，后一项把前一项的下降补了回来。
//
// 与 v1_naive 的关系
//   v1 也已经合并访存 —— 它的问题是**并发度不足**，不是访存模式不好。
//   所以 v2 修的是另一个瓶颈：每线程只有 1 个访存在飞。
//
// 代价
//   寄存器占用上升（要同时装下 4 个 x 和 4 个 y），ILP 与 occupancy 在这里
//   开始互相拉扯。这正是本章值得动手调 kUnroll 的地方。
//
// 预期
//   有效带宽比 v1 高约 10 个百分点。若实测没提升甚至下降，说明 occupancy
//   已经被寄存器吃掉了 —— 那也是一个有价值的结论。
// ---------------------------------------------------------------------------
void add_v2_unroll4(const float* x, const float* y, float* out, int64_t n, cudaStream_t stream);
void relu_v2_unroll4(const float* x, float* out, int64_t n, cudaStream_t stream);
void sigmoid_v2_unroll4(const float* x, float* out, int64_t n, cudaStream_t stream);

// ---------------------------------------------------------------------------
// v3_grid_stride —— 固定 grid + kernel 内循环步进
//
// 思路
//   grid = SM 数 × 每 SM 能驻留的 block 数（"刚好填满机器"），每个线程靠
//   循环步进处理 i、i+stride、i+2*stride……。grid 不再随 n 增长。
//
// 确定的收益（功能性）
//   v1 / v2 的 grid 受 2^31-1 上限约束，超出就抛错；v3 的循环步进天然覆盖
//   任意大的 n，**没有上限**。另外循环条件本身就是完整的边界处理，
//   不需要额外的尾部路径。
//
// 不确定的收益（性能）
//   "消除尾部空转"这个流传很广的理由，对大规模输入其实很弱：n = 2^24 时
//   grid 有 65536 个 block 而机器一次只能驻留一百多个，最后那个没填满的 wave
//   只占百分之零点几。
//   所以 v3 可能和 v1 / v2 差不多快，甚至更慢。**这要靠实测判断** ——
//   三者带宽相当本身就是一个有价值的结论。
// ---------------------------------------------------------------------------
void add_v3_grid_stride(const float* x, const float* y, float* out, int64_t n,
                        cudaStream_t stream);
void relu_v3_grid_stride(const float* x, float* out, int64_t n, cudaStream_t stream);
void sigmoid_v3_grid_stride(const float* x, float* out, int64_t n, cudaStream_t stream);

// ---------------------------------------------------------------------------
// v4_vectorized —— float4 向量化
//
// 思路
//   一次访存搬 16 字节而不是 4 字节，**访存指令数降 4 倍**。
//   刻意不加 grid-stride：v4 = v1 + 向量化，只改一个变量，这样阶梯表里
//   v1 → v4 的差距才能干净地归因于访存宽度。
//
// 四个边界风险（本章唯一真正容易写错的地方）
//   1. n 不是 4 的倍数       → 尾部 0~3 个元素走标量路径
//   2. 指针不是 16 字节对齐   → **退回 v1 的标量路径，而不是报错**
//   3. n < 4                → n4 = 0，但要保证 grid 至少有 1 个 block，
//                             否则尾部无人处理
//   4. grid 覆盖不到尾巴      → 尾部复用线程 0~2 的下标
//
// 预期
//   比 v1 高约 10 个百分点。若实测几乎没差别，说明这台机器在 v1 那种写法下
//   带宽就已经接近饱和，向量化省下的是指令而不是时间。
// ---------------------------------------------------------------------------
void add_v4_vectorized(const float* x, const float* y, float* out, int64_t n,
                       cudaStream_t stream);
void relu_v4_vectorized(const float* x, float* out, int64_t n, cudaStream_t stream);
void sigmoid_v4_vectorized(const float* x, float* out, int64_t n, cudaStream_t stream);

}  // namespace elementwise
}  // namespace ops_lab
