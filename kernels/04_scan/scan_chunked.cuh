// kernels/04_scan/scan_chunked.cuh
//
// **两级**块内扫描：每线程处理 K 个元素，把 per-chunk 的固定开销摊销 K 倍。
//
// ## 为什么需要它（第 4 章第一次实测暴露出来的问题）
//
// 第一版每个线程只处理 1 个元素（chunk = 256），于是每个 chunk 要付
// 8~18 次 `__syncthreads()` —— 也就是**每元素 0.09 次 barrier**。
// 实测代价很清楚：v1 是 727.8 µs，而它的 12N 流量（101.2 MB）按实际带宽只值 ~427 µs，
// 差的 300 µs 里绝大部分就是这些 barrier 和 smem 往返。
//
// 标准解法（也是 CUB 的做法）是把 chunk 放大到 `kBlockSize × K`，结构变成三级：
//
//     ① 合并载入 → smem（striped 读，blocked 带 padding 写）
//     ② 每线程**串行**扫自己那 K 个连续元素（寄存器里，1 次加法/元素）
//     ③ 256 个 thread total 做一次块级扫描 ← 策略（朴素/KS/Blelloch）在这里起作用
//     ④ 加回 thread 级前缀，再经 smem 转置成 striped，**合并写回**
//
// 这样 barrier 次数不变（还是 8~18 次），但服务的元素多了 K 倍：
// **每元素 barrier 从 0.09 降到 0.011（K=8）**。
//
// ## 为什么必须转置（这一步最容易写错）
//
// 合并载入要求 thread t 读 x[base + k·B + t]（striped）；
// 而"每线程串行扫连续 K 个"要求 thread t 拥有 x[base + t·K .. +K)（blocked）。
// 两者不一致，所以要在 smem 里**转置**一次。写回时再转回来（否则写会变成
// 跨步 → 一个 warp 32 条 cache line，直接掉回第 2 章 v0 的下场）。
//
// ## padding 的作用
//
// blocked 布局的行长取 K+1（而不是 K）：读 `smem[t·(K+1) + k]` 时，
// 同一时刻各线程的地址是 `t·(K+1) + k`，模 32 两两不同（K+1 与 32 互质）→ 无 bank conflict。
// 用 K 就会退化成 K 路冲突。
#pragma once

#include "kernels/04_scan/scan_block.cuh"

namespace ops_lab {
namespace scan {

// 两级扫描需要的 smem scratch（float 个数）：
//   blocked 带 padding 的缓冲区 kBlockSize*(K+1)
//   + 1 个槽用来广播 block 前缀
// 块级策略用的那 kBlockSize+1 个 float 与原缓冲区复用（有时序保证，见下面的屏障）。
template <int K>
constexpr int chunked_scratch_floats() {
  return kBlockSize * (K + 1) + 1;
}

// 前缀广播槽的下标
template <int K>
constexpr int chunked_prefix_slot() {
  return kBlockSize * (K + 1);
}

// ---------------------------------------------------------------------------
// 第一级 + 第二级：把 [base, base + kBlockSize*K) 扫成"块内 exclusive"，
// 结果通过 `local[k]`（本线程的 K 个元素）与 `t_exclusive`（本线程之前的全部和）返回，
// chunk 总和由 `total` 返回。
//
// 注意这一级**不**写全局内存 —— 单趟实现要先把总和发布出去、拿到 block 前缀，
// 才轮到写回。所以扫描与写回必须拆成两步（见 chunk_store）。
// ---------------------------------------------------------------------------
template <typename Policy, int K>
__device__ __forceinline__ void chunk_scan_two_level(const float* __restrict__ x, int64_t base,
                                                    int64_t n, float* smem, float (&local)[K],
                                                    float& t_exclusive, float& total) {
  // ① 合并载入 + 转置成 blocked 布局（带 padding）
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int j = k * kBlockSize + static_cast<int>(threadIdx.x);
    const int64_t gi = base + j;
    const float v = (gi < n) ? x[gi] : 0.0f;
    smem[(j / K) * (K + 1) + (j % K)] = v;
  }
  __syncthreads();

  // ② 每线程串行扫自己那 K 个连续元素（寄存器里，1 次加法/元素）
  float acc = 0.0f;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    acc += smem[static_cast<int>(threadIdx.x) * (K + 1) + k];
    local[k] = acc;  // 本线程的 inclusive 前缀
  }
  const float t_total = acc;  // ← 注意：这是**本线程** K 个元素的和，不是 chunk 总和

  // 读完 blocked 数据才能复用 smem（否则会与下面写 smem[tid] 撞车）
  __syncthreads();

  // ③ 256 个 thread total 做一次块级扫描 —— 策略在这里起作用。
  //
  // **返回值才是 chunk 总和**（256 个 t_total 的和）。
  // 这里踩过一次：一度把 t_total 当成 chunk 总和发布出去，于是全 1 输入下
  // chunk 0 发布的是 x[0] = 1 而不是 256 —— 从第二个 chunk 开始全错。
  // 边界测试里那条"全 1 输入要求逐位相等"就是为了抓这类错（随机输入的容差会吃掉它）。
  smem[threadIdx.x] = t_total;
  total = Policy::scan(smem, t_total);  // 出 smem 前 kBlockSize 项已经变成 exclusive
  t_exclusive = smem[threadIdx.x];

  // 策略用完 smem，才能拿它当转置缓冲区（下面 chunk_store 会重写它）
  __syncthreads();
}

// ---------------------------------------------------------------------------
// 第三级：加上 block 前缀，经 smem 转置回 striped，合并写回全局。
// ---------------------------------------------------------------------------
template <int K>
__device__ __forceinline__ void chunk_store(float* __restrict__ out, int64_t base, int64_t n,
                                           float* smem, const float (&local)[K],
                                           float t_exclusive, float block_prefix) {
  const float addend = t_exclusive + block_prefix;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    smem[static_cast<int>(threadIdx.x) * (K + 1) + k] = addend + local[k];
  }
  __syncthreads();

  // striped 读 + 合并写（这里是整个 kernel 唯一的全局写）
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int j = k * kBlockSize + static_cast<int>(threadIdx.x);
    const int64_t gi = base + j;
    if (gi < n) {
      out[gi] = smem[(j / K) * (K + 1) + (j % K)];
    }
  }
}

}  // namespace scan
}  // namespace ops_lab
