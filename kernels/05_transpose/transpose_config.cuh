// kernels/05_transpose/transpose_config.cuh
//
// 第 5 章 transpose 的尺寸常量。**只有常量，没有 device 代码**
// （绑定层的 .cpp 会 include transpose.cuh，不能连带把 device 代码拖进去）。
//
// ## 为什么瓦片是 32×32 起
//
// bank 数是 32，一个 float 占 4 字节 —— 所以"一行 32 个 float"恰好铺满所有 bank。
// 这既是最自然的瓦片宽度，也正好是**列访问会全撞同一个 bank** 的宽度
// （`tile[tx][ty]` 的地址是 `tx*32 + ty`，一个 warp 的 32 个线程算出来全是同一 bank）。
// 本章的全部张力就来自这 32。
#pragma once

#include <cstdint>

namespace ops_lab {
namespace transpose {

// ---------------------------------------------------------------- 经典形态
//
// block = (32, 8) = 256 线程，瓦片 32×32，每个线程负责 4 行（跨步 8）。
// 256 线程 × 6 block = 1536 = 每 SM 线程上限 → 100% occupancy。
constexpr int kTileRows = 32;
constexpr int kTileCols = 32;
constexpr int kBlockRows = 8;   // block.y
constexpr int kBlockCols = 32;  // block.x
constexpr int kBlockSize = kBlockRows * kBlockCols;  // 256

// ------------------------------------------------- v0 / v1（不用 smem）
//
// 一线程一元素，block = (32, 32) = 1024。两个变体只差"相邻线程沿哪个下标走"。
constexpr int kNaiveBlock = 32;

// ------------------------------------------------------------ 向量化形态
//
// block = (8, 32) = 256 线程；瓦片 32×32：
//   x 方向 8 个线程 × 4 个连续列 = 32 列（float4）
//   y 方向 32 个线程 = 32 行
// 这样读侧和写侧都能用 float4，而 bank 冲突依旧由 padding 解决。
constexpr int kVecBlockCols = 8;
constexpr int kVecBlockRows = 32;
constexpr int kVecItems = 4;  // 每线程 4 个元素（一个 float4）
static_assert(kVecBlockCols * kVecItems == kTileCols, "float4 要恰好铺满瓦片宽度");

// --------------------------------------------------------------- 宽瓦片
//
// block 仍是 (8, 32) = 256 线程，但瓦片加宽到 32×64：
// 每个 block 每次连续访问的字节数从 128 B 涨到 256 B。
//
// 这一级存在的理由：**DRAM 的突发局部性**。32 列 = 128 B 恰好是一个 sector 段，
// 而 64 列 = 256 B；行缓冲命中率会跟着变好。这是本章唯一"可能还能继续爬"的一级。
constexpr int kWideTileCols = 64;
constexpr int kWideVecLoops = kWideTileCols / (kVecBlockCols * kVecItems);  // 2 次

// --------------------------------------------------------------- padding
//
// 行宽 +1：列访问的地址变成 `tx*(W+1) + ty`，模 32 之后两两不同（W+1 与 32 互质）。
// 这是"消 bank conflict"最便宜的一招。
constexpr int kPad = 1;

// 向量化的那两个变体**也用 +1**：共享内存仍然用标量读写（只有全局访存是 float4），
// 所以 +1 就够了 —— 见 transpose_v4_vectorized.cu 里的 bank 推算。
//
// **为什么这里值得写一句**：如果共享内存也要 float4 读写，行宽就必须是 4 的倍数
// （否则第二行的 float4 会跨在不对齐的位置），而 4 的倍数与 32 不互质
// （+4 时 bank 偏移变成 4·行，8 行就绕回原处）→ padding 这一招在向量化场景下失效。
// 真实实现（CUTLASS 等）这时改用 **XOR swizzle**。本章不做，因为它会把
// "瓦片宽度"这个变量污染掉 —— 但这条边界值得记住。
constexpr int kVecPad = 1;

// workspace：本章**不需要**（转置没有跨 block 的依赖）
constexpr int64_t workspace_floats(int64_t /*n*/) { return 0; }

}  // namespace transpose
}  // namespace ops_lab
