// kernels/05_transpose/transpose.cuh
//
// 第 5 章 transpose 的 host launcher 声明。
//
// 铁律不变：不出现任何 torch 符号；也**不包含任何带 __device__ 的头**
// （绑定层的 .cpp 会 include 它，见 docs/DESIGN.md §5.1）。
//
// ============================ 本章的问题：out[c][r] = in[r][c] ============================
//
// M×N 的矩阵转成 N×M。**输出形状与输入不同**（第 3 章加的 `out_shape` 元数据字段
// 正好用在这里）。
//
// 搬运量还是 8MN（读 4MN + 写 4MN，与第 2、4 章同一个访存形态），
// 但转置有它自己的麻烦：
//
//   **总有一边是跨步的。** 要么读跨步、要么写跨步 —— 除非用共享内存把另一边
//   先"摆正"。这一条是整章的全部内容。
//
// ---------------------------------------------------------------- 先算一笔账
//
// 教科书说：32×32 的 smem 瓦片，列访问会造成 **32 路 bank conflict**，
// 要付 2~4 倍的代价；padding 一行就能消掉。**这笔账在这台机器上要重算。**
//
// 关键在于**共享内存带宽比显存带宽快多少**：
//
//     smem:  32 bank/SM × 24 SM = 768 次 bank 访问/cycle
//     DRAM:  256 GB/s ÷ 8 B/元素 = 32e9 元素/s ≈ 16 元素/cycle（按 2 GHz）
//
//     每元素需要的 smem 访问次数：
//         无冲突   写 1 + 读 1        = 2 次  → 768/2  = 384 元素/cycle
//         32 路冲突 写 1 + 读 32       = 33 次 → 768/33 =  23 元素/cycle
//
// 于是：
//
//     * 无冲突时 smem 能喂 **384** 元素/cycle，DRAM 只能吃 16 —— smem 快 24 倍
//     * 冲突之后 smem 掉到 **23**，仍然高于 DRAM 的 16，但**只快 1.4 倍了**
//
// 所以预测不是"慢 2~4 倍"，而是：
//
//     **v3_padded 比 v2_smem_tiled 快 5~25%** —— 冲突会露出来一点，但远不到教科书的量级。
//
// 若实测差出 2 倍以上，说明上面这个比值算错了（那也值得记下来）。
// 而若实测差不到 5%，说明连"23 > 16"这点余量都被掩盖了。
//
// **这一章和第 3 章的对照关系是刻意的**：第 3 章的 v2（padding）是**真·空操作**，
// 因为顺序寻址本来就没有冲突；这里的冲突是**真实存在**的。两章合起来才能说清
// "经典优化什么时候有用"。
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "kernels/05_transpose/transpose_config.cuh"
#include "kernels/common/cuda_check.h"
#include "kernels/common/device.h"

namespace ops_lab {
namespace transpose {

// v0/v1 用 (32,32) 的二维网格铺满矩阵
inline dim3 naive_grid(int rows, int cols) {
  return dim3(static_cast<unsigned int>(ceil_div(cols, kNaiveBlock)),
              static_cast<unsigned int>(ceil_div(rows, kNaiveBlock)));
}

// v2~v5 按 32×32 的瓦片切（宽瓦片那一级只改 block 内部的列数，网格不变）
inline dim3 tiled_grid(int rows, int cols) {
  return dim3(static_cast<unsigned int>(ceil_div(cols, kTileCols)),
              static_cast<unsigned int>(ceil_div(rows, kTileRows)));
}

// ---------------------------------------------------------------------------
// v0_naive —— 反面教材之一：读合并、**写跨步**
//
//   线程 (tx, ty) 处理 in[r][c]：相邻 tx → 相邻 c → 读天然合并；
//   而 out[c*rows + r] 的相邻 tx 之间隔了 `rows` 个元素 → 一个 warp 落在 32 条
//   cache line 上。
//
//   预期：明显慢，**但不是教科书的 32 倍**。第 2 章的教训在这里同样适用：
//   写是"回写"的，L2 能把同一个 sector 的多次写入合并起来，所以代价是
//   **事务数**而不是显存带宽。
// ---------------------------------------------------------------------------
void transpose_v0_naive(const float* in, float* out, int rows, int cols, cudaStream_t stream);

// ---------------------------------------------------------------------------
// v1_swapped —— 反面教材之二：**读跨步**、写合并
//
//   与 v0 **只差交换 (r, c) 的取法**：同样的一个 kernel，让相邻线程沿 r 走。
//
//   预期：**比 v0 更差**。理由是不对称的 ——
//   读的数据要立刻拿到才能算，延迟藏不住；写只要进了回写缓冲就算了事。
//   这一级存在的意义就是把"读写不对称"这件事量出来。
// ---------------------------------------------------------------------------
void transpose_v1_swapped(const float* in, float* out, int rows, int cols, cudaStream_t stream);

// ---------------------------------------------------------------------------
// v2_smem_tiled —— 引入 32×32 的共享内存瓦片，两边都合并
//
//   读：`tile[ty][tx] = in[...]`（相邻 tx 相邻地址，合并；写 smem 行 → 无冲突）
//   写：`out[...] = tile[tx][ty]`（写全局合并；**读 smem 列 → 32 路冲突**）
//
//   预期：相对 v1 是**一大步**（两边都合并了），但还留着一个 32 路的 bank conflict。
// ---------------------------------------------------------------------------
void transpose_v2_smem_tiled(const float* in, float* out, int rows, int cols,
                             cudaStream_t stream);

// ---------------------------------------------------------------------------
// v3_padded —— 与 v2 **只差共享内存的行宽**（32 → 33）
//
//   列访问的地址从 `tx*32 + ty` 变成 `tx*33 + ty`，模 32 之后两两不同 → 无冲突。
//
//   预期：**快 5~25%**（见文件头那笔账），不是教科书里的 2~4 倍。
// ---------------------------------------------------------------------------
void transpose_v3_padded(const float* in, float* out, int rows, int cols, cudaStream_t stream);

// ---------------------------------------------------------------------------
// v4_vectorized —— 与 v3 只差**访存宽度**：读写两侧都用 float4
//
//   block 改成 (8, 32)：x 方向 8 个线程 × 4 个连续列 = 32 列。
//   注意线程映射**必须**跟着改 —— float4 要求每个线程覆盖 4 个连续元素，
//   这是向量化的固有代价，不是疏忽（第 2 章的 v4 也有同样的映射变化）。
//
//   预期：省指令数；**可能和 v3 打平甚至更慢** —— 第 2 章的 v4 就是更慢的那个。
// ---------------------------------------------------------------------------
void transpose_v4_vectorized(const float* in, float* out, int rows, int cols,
                             cudaStream_t stream);

// ---------------------------------------------------------------------------
// v5_wide_tile —— 与 v4 只差**瓦片宽度**：32×32 → 32×64
//
//   每个 block 每次连续访问从 128 B 变成 256 B → **DRAM 突发局部性变好**。
//   这是本章唯一"可能还能继续往上爬"的一级（和第 1 章 `probe_smem48k`
//   那个未解释现象同源：并发流少一点、连续段长一点，显存效率就高一点）。
//
//   预期：比 v4 好 5~15%。**不确定，而且很可能测不出来** —— 那就如实记下来。
// ---------------------------------------------------------------------------
void transpose_v5_wide_tile(const float* in, float* out, int rows, int cols,
                            cudaStream_t stream);

}  // namespace transpose
}  // namespace ops_lab
