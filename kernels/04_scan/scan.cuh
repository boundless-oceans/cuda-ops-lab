// kernels/04_scan/scan.cuh
//
// 第 4 章 scan（inclusive prefix sum）的 host launcher 声明。
//
// 铁律不变：这个头文件以及所有 .cu 里**不出现任何 torch 符号**。
// 也**不包含任何带 __device__ 的头**（绑定层的 .cpp 会 include 它）——
// 这个坑第 3 章踩过，见 docs/DESIGN.md §5.1。
//
// ============================ 本章的问题：y[i] = Σ_{j≤i} x[j] ============================
//
// 和第 3 章（归约）一样要处理"元素之间有序依赖"，但结论**完全相反**。
// 先把两笔账算出来，因为整章的设计都建立在这两笔账上。
//
// ---------------------------------------------------------------- 第一笔账：块内 work
//
// 第 3 章算出"块内树只占理论下限的 0.1%，所以怎么优化都测不出来"。
// 那个结论的依据是：reduction 里**每个元素只参与一次合并** → 块内 work 是 O(1)/元素。
//
// scan 不是。每个元素要参与多少工作，取决于块内算法：
//
//     朴素        每元素 O(B)      一个 256 的 chunk 要 B(B−1)/2 ≈ 3.3e4 次 smem 访问
//     Kogge-Stone 每元素 O(log B)   8 级 × 256 线程 × 2 次访问 ≈ 4.6e3
//     Blelloch    每元素 O(1) 摊销  上扫 255 + 下扫 255，约 1.3e3
//
// N = 2²³ 时有 C = 32768 个 chunk。sm_89 的 smem 吞吐约
// 32 个 float/cycle/SM × 24 SM = 768/cycle（约 2 GHz → 1.5e12 次/秒）：
//
//     朴素         1.07e9 次 → **0.71 ms**
//     Kogge-Stone  1.5e8 次  → 0.10 ms
//     Blelloch     4.2e7 次  → 0.028 ms
//
// 而内存时间是 **262 µs**。所以：
//
//     **朴素扫描的块内 work 是内存时间的 2.7 倍 —— 它会成为瓶颈。**
//
// 这是第 4 章和第 3 章最大的差别，也是本章要验证的核心预测：
// **"块内优化没用"这条结论不能外推**，它依赖"块内 work 是 O(1)/元素"这个前提。
//
// ------------------------------------------------------------- 第二笔账：全局流量
//
// 多 block 的 scan 必须先知道"前面所有 chunk 的和"。三段式（scan-scan-add）的做法是
// 读两遍 x：
//
//     A 趟 读 x(4N) → 每 chunk 一个和        ┐
//     B 趟 扫这些和（数据量 N/256，可忽略）   ├─ 共 12N = 100.7 MB → 下限 393 µs
//     C 趟 再读 x(4N) + 写 out(4N)           ┘
//
// 而理想是读一遍写一遍 = 8N = 67.1 MB → 262 µs。**多读一遍输入，代价 1.5 倍。**
//
// 第 3 章的多 block 合并是 144 次原子操作，**免费**；scan 的合并第一次要付真实流量。
// 要把 12N 降到 8N，就必须让 C 趟在写回之前就拿到自己的前缀 —— 也就是
// `scan_v3_single_pass.cu` 里的 decoupled look-back。
//
// 这件事带来一个**很容易踩的坑**：`%峰值` 的分母到底该用 8N 还是 12N？
// 用同一个就是拿两把尺子量东西。所以本章的元数据给每个变体声明**自己的 bytes**，
// 表里同时给出"实际流量"和"按各自实际流量算的 %峰值"。
// 于是会读出一个新结论：**`%峰值` 高不等于快** —— v1/v2 把自己的 12N 用到 93%，
// 仍然比把 8N 用到 95% 的 v3 慢 45%。
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "kernels/04_scan/scan_config.cuh"
#include "kernels/common/cuda_check.h"
#include "kernels/common/device.h"

namespace ops_lab {
namespace scan {

// ------------------------------------------------------------ grid 计算
//
// v0~v2 用"一个 chunk 一个 block"，所以 grid = chunk 数。
// 上限检查复用第 2 章的 grid_for（超出 2^31−1 会抛清晰错误）。
inline unsigned int grid_for_chunks(int64_t n) {
  return static_cast<unsigned int>(grid_for(chunk_count(n), 1, /*allow_grid_stride=*/false));
}

// v3 的 grid 必须是"填满机器"的规模，且**不得超过可同时驻留的 block 数**。
inline int64_t capped_grid(int64_t n) {
  const int64_t cap = default_grid_size(kBlockSize);
  const int64_t need = chunk_count(n);
  return need < cap ? need : cap;
}

// ---------------------------------------------------------------------------
// v3 的死锁护栏 —— 这一条不是洁癖，是保命
//
// decoupled look-back 让每个 block **自旋等待**前面的 chunk 发布前缀。
// 这要求所有 block 同时驻留：只要有一个 block 还没被调度上来，它前面的 block
// 就可能永远等下去 → **整个 GPU 挂死**（而不是报错）。
//
// 在笔记本这种"显卡同时驱动桌面"的场景里，挂死非常难受。所以这里做两件事：
//   1. grid 上限取 `default_grid_size()`（= SM 数 × 每 SM 可驻留 block 数），
//      这是**恰好等于**可同时驻留容量的值 —— 这就是它安全的原因；
//   2. 万一有人把 grid 调大，这里**直接抛异常**，而不是让它去挂死。
inline void require_co_resident(int64_t grid, int64_t n) {
  const int64_t cap = default_grid_size(kBlockSize);
  if (grid > cap) {
    throw std::runtime_error(
        "scan 单趟实现要求所有 block 同时驻留，否则会死锁：\n"
        "  chunk 数需要 " + std::to_string(grid) + " 个 block，而本机最多同时驻留 " +
        std::to_string(cap) + " 个（SM 数 × 每 SM 块数）。\n"
        "  n = " + std::to_string(n) + "。请改用三段式变体（v0~v2），"
        "或者把 grid 降到可驻留容量以内。");
  }
}

// ---------------------------------------------------------------------------
// v0_naive —— 反面教材：块内用 O(B²) 的朴素扫描
//
//   思路：线程 t 把自己前面 t 个元素加一遍。正确，但 work 是 O(N·B)。
//   预期：**块内 work（约 0.71 ms）超过内存时间（262 µs）** → 约 25~30% 峰值。
//   它存在的意义：证明"work 效率"在 scan 上是**真的**会变成时间的，
//   而在 reduction 上不会（那里每元素只有 O(1) 次合并）。
// ---------------------------------------------------------------------------
void scan_inclusive_v0_naive(const float* x, float* out, int64_t n, float* workspace,
                             cudaStream_t stream);

// ---------------------------------------------------------------------------
// v1_kogge_stone —— 把块内 work 降到 O(N log B)
//
//   与 v0 只差一行（第三趟用哪个策略）。
//   预期：块内 0.10 ms 已低于内存时间 → 应该回到内存受限，~85~90% 峰值
//   （按 12N 的实际流量算）。
// ---------------------------------------------------------------------------
void scan_inclusive_v1_kogge_stone(const float* x, float* out, int64_t n, float* workspace,
                                   cudaStream_t stream);

// ---------------------------------------------------------------------------
// v2_blelloch —— work-efficient（每元素 O(1) 摊销）
//
//   与 v1 只差一行：块内换成上扫+下扫。级数从 8 涨到 16，但 work 少了约 3 倍。
//   预期：块内 0.028 ms 完全被掩盖 → **与 v1 几乎一样快**（都是 12N 的
//   内存下限 393 µs 附近）。
//   若实测 v2 明显快于 v1，说明"少做 work"在块内 work 尚未被完全掩盖时仍然值钱。
// ---------------------------------------------------------------------------
void scan_inclusive_v2_blelloch(const float* x, float* out, int64_t n, float* workspace,
                                cudaStream_t stream);

// ---------------------------------------------------------------------------
// v3_single_pass —— 单趟 decoupled look-back（本章的压舱石）
//
//   块内沿用 v2 的 Blelloch，只改**多 block 怎么合并**：
//   不再"读两遍 x"，而是每个 block 处理一个 chunk 时，把 chunk 的和发布到
//   workspace，然后**向前看**（look back）累加前面 chunk 已发布的和，
//   拿到自己的 exclusive 前缀后立刻写回。
//   全局流量因此从 12N 降到 **8N**。
//
//   代价与约束：
//     * block 必须能同时驻留（见 require_co_resident）
//     * 需要 `cudaMemsetAsync` 把 status 数组清零（多一次流操作，~1~2 µs）
//     * 代码里第一次出现 `__threadfence()` 和自旋等待
//
//   预期（第一版实测）：**1874.9 µs / 14.1%** —— 差了 6.8 倍，原因不是算法错，
//   而是回看被写成了**单线程**（只让 thread 0 逐个往前找）。同一轮 144 个 block 是
//   相邻 chunk，回看距离最长 144 步 = 144 次全局往返。
//   **"少读一遍输入"这笔账是对的，但少了"整 warp 并行回看"这个必要件。**
//   修法见 scan_lookback.cuh。
// ---------------------------------------------------------------------------
void scan_inclusive_v3_single_pass(const float* x, float* out, int64_t n, float* workspace,
                                   cudaStream_t stream);

// ---------------------------------------------------------------------------
// v4_optimized —— 把两处修法合起来（本章的答案）
//
//   结构沿用 v3 的单趟 look-back（8N）+ 修好的整 warp 并行回看，
//   再把**每线程元素数**从 1 提到 8（chunk 256 → 2048）。
//   代码上与 v3 的差别只有一个模板参数（见 scan_single_pass.cuh）。
//
//   为什么要摊销是实测逼出来的：第一版 v1 的 12N 流量只值 427 µs、实测 727.8 µs，
//   差的 300 µs 里绝大部分是 per-chunk 的 barrier 与 smem 往返（每元素 0.031 次
//   barrier）。放大 chunk 之后这项除以 8。见 scan_chunked.cuh。
//
//   预期：**~85~90% 峰值（按 8N 算），约 300~330 µs** —— 也就是 torch 现在的位置
//   （实测 torch 294.9 µs / 88.9%）。能不能打平是本级唯一的判据。
// ---------------------------------------------------------------------------
void scan_inclusive_v4_optimized(const float* x, float* out, int64_t n, float* workspace,
                                 cudaStream_t stream);

// ---------------------------------------------------------------------------
// v5_single_block —— 故意回退：整个数组交给一个 block
//
//   块内沿用本章最省事的那套（Blelloch，每线程一个元素），但 grid = 1，
//   靠一个 block 顺序走过所有 chunk（用一个 carry 串起来）。不需要 workspace。
//
//   预期（第一版实测）：**31098.9 µs / 0.8% 峰值**。
//   31 ms ÷ 32768 chunk ≈ 0.95 µs/chunk ≈ 18 次 barrier 的串行开销。
//
//   它和 v4 的对比是本章第二个数量级教训：结构、块内算法、摊销都选对之后，
//   **只用一个 block 依然慢两个数量级**。
//   （第 3 章有同一个教训：那里的 v4_single_block 慢 16.3 倍。）
// ---------------------------------------------------------------------------
void scan_inclusive_v5_single_block(const float* x, float* out, int64_t n, cudaStream_t stream);

}  // namespace scan
}  // namespace ops_lab
