// kernels/04_scan/scan_block.cuh
//
// 块内扫描的三种写法 —— 第 4 章真正要比的东西。
//
// ## 为什么这一章要比"块内"（第 3 章刚说过块内不重要）
//
// 因为 scan 的块内 work **不是 O(1)**。reduction 里每个元素只参与一次合并，
// 所以块内那棵树只占理论下限的 0.1%；scan 里每个元素要参与 log B 次（甚至 B 次），
// 于是块内 work 第一次可以**超过内存时间**。先算这笔账（见 scan.cuh）再写代码。
//
// ## 统一契约（三个函数完全一致，这样变体之间只差一行）
//
//   * 入口：`smem[0..kBlockSize)` 是本 chunk 的元素（尾部不足的用 **0** 补齐），
//     `v` 是本线程负责的那个元素（尾部补 0 的线程 v = 0）
//   * 出口：smem[0..kBlockSize) **就地变成 exclusive 前缀和**，
//     返回值（所有线程都拿到）= 本 chunk 的总和
//
// 于是调用方的写回永远是同一行：
//
//     if (gi < n) out[gi] = smem[threadIdx.x] + v + block_prefix[blockIdx.x];
//
// （exclusive 前缀 + 自己 + 前面所有 chunk 的前缀 = inclusive 结果）
#pragma once

#include "kernels/04_scan/scan_config.cuh"

namespace ops_lab {
namespace scan {

// ------------------------------------------------------- 归约（第 A 趟用）
//
// 三段式的第一趟要把每个 chunk 缩成一个和。这里直接用第 3 章那棵
// **顺序寻址的 smem 树**（那一章已经证明它够快，而且没有 bank conflict）。
template <int kBlockSize>
__device__ __forceinline__ float block_reduce_sum(float* smem) {
  for (int s = kBlockSize >> 1; s > 0; s >>= 1) {
    __syncthreads();  // 同步在读之前（第 3 章的教训：写在循环末尾是竞态）
    if (static_cast<int>(threadIdx.x) < s) {
      smem[threadIdx.x] += smem[threadIdx.x + s];
    }
  }
  __syncthreads();
  return smem[0];
}

// ------------------------------------------------------------ ① 朴素扫描
//
// 线程 t 把自己前面 t 个元素加一遍。**这是本章的反面教材**：
// 每个 chunk 的 work 是 B(B+1)/2 次加法，也就是每元素 O(B)、全数组 O(N·B)。
// 第 3 章的"每元素 O(1)"在这里变成了 O(B) —— 差了 256 倍。
//
// 教科书上的朴素 scan 是"每个输出元素把全数组前缀加一遍"（O(N²)，在本仓库的
// 规模下要跑几十分钟，根本没法测）。这里的版本把它限制在 chunk 内，
// 是同一个错误的**可测量版本**：work 爆炸的机理完全一样。
template <int kBlockSize>
__device__ __forceinline__ float naive_block_scan_exclusive(float* smem, float v) {
  __syncthreads();
  const int t = static_cast<int>(threadIdx.x);
  float acc = 0.0f;
  for (int j = 0; j < t; ++j) {
    acc += smem[j];
  }
  // 读阶段结束才能写：否则会读到别的线程写了一半的值
  __syncthreads();
  smem[t] = acc;
  // chunk 总和 = 最后一个线程的 exclusive 前缀 + 它自己那个元素
  if (t == kBlockSize - 1) {
    smem[kBlockSize] = acc + v;
  }
  __syncthreads();
  return smem[kBlockSize];
}

// ------------------------------------------------------- ② Kogge-Stone
//
// 8 级，每级让每个线程把自己前面 2^k 个位置的值加进来。
// work = O(N log B)：比 Blelloch 多 log B 倍，但级数只有 log B（深度浅）。
//
// 两级之间的 `__syncthreads()` 位置很关键：**先把要读的值读进寄存器，
// 再同步，最后才写**。反过来的话，同一级里后面的线程会读到本轮已被覆盖的值。
template <int kBlockSize>
__device__ __forceinline__ float kogge_stone_block_scan_exclusive(float* smem, float /*v*/) {
  __syncthreads();
  for (int offset = 1; offset < kBlockSize; offset <<= 1) {
    const int t = static_cast<int>(threadIdx.x);
    const float prev = (t >= offset) ? smem[t - offset] : 0.0f;
    __syncthreads();  // 读完之后才允许写
    smem[t] += prev;
    __syncthreads();
  }
  // smem 现在是 inclusive 前缀；先取走总和，再整体右移一位变成 exclusive
  const float total = smem[kBlockSize - 1];
  const int t = static_cast<int>(threadIdx.x);
  const float shifted = (t > 0) ? smem[t - 1] : 0.0f;
  __syncthreads();  // 保证所有线程都取到了 total 和 shifted
  smem[t] = shifted;
  return total;
}

// --------------------------------------------------------- ③ Blelloch
//
// work-efficient：上扫 + 下扫，一共约 2(B−1) 次加法（每元素 O(1) 摊销），
// 代价是级数翻倍（16 级 vs 8 级）。
//
// 这是教科书里"work-efficient scan"的标准形态，也是本章要验证的东西：
// **在一台 memory-bound 的机器上，"少做 work"到底值不值？**
template <int kBlockSize>
__device__ __forceinline__ float blelloch_block_scan_exclusive(float* smem, float /*v*/) {
  __syncthreads();

  // 上扫：把相邻两段的 partial sum 往上推，根节点 = 总和
  for (int stride = 1; stride < kBlockSize; stride <<= 1) {
    const int idx = (static_cast<int>(threadIdx.x) + 1) * (stride << 1) - 1;
    if (idx < kBlockSize) {
      smem[idx] += smem[idx - stride];
    }
    __syncthreads();  // 本级的写入要在下一级读之前可见
  }

  // 根就是总和。先让所有线程取走，再把它置 0 —— 置 0 与读取之间必须有屏障，
  // 否则别的线程可能读到 0 而不是总和。
  const float total = smem[kBlockSize - 1];
  __syncthreads();
  if (threadIdx.x == 0) {
    smem[kBlockSize - 1] = 0.0f;
  }

  // 下扫：把根的值往下传播，得到 exclusive 前缀
  // 同一级里各线程读写的下标互不相交（idx 与 idx-stride 都是唯一的一段），
  // 所以每级只需要一次屏障（放在读之前）。
  for (int stride = kBlockSize >> 1; stride > 0; stride >>= 1) {
    __syncthreads();
    const int idx = (static_cast<int>(threadIdx.x) + 1) * (stride << 1) - 1;
    if (idx < kBlockSize) {
      const float left = smem[idx - stride];
      smem[idx - stride] = smem[idx];
      smem[idx] += left;
    }
  }
  __syncthreads();
  return total;
}

// ============================================================ 三个策略包装
//
// 让"变体之间的差别"变成**一行模板参数**，而不是三份复制粘贴的 kernel。
// 三个调用点的签名完全一致，所以 scan_apply.cu 里那份 kernel 只写一遍。
struct NaiveScanPolicy {
  static __device__ __forceinline__ float scan(float* smem, float v) {
    return naive_block_scan_exclusive<kBlockSize>(smem, v);
  }
};

struct KoggeStoneScanPolicy {
  static __device__ __forceinline__ float scan(float* smem, float v) {
    return kogge_stone_block_scan_exclusive<kBlockSize>(smem, v);
  }
};

struct BlellochScanPolicy {
  static __device__ __forceinline__ float scan(float* smem, float v) {
    return blelloch_block_scan_exclusive<kBlockSize>(smem, v);
  }
};

}  // namespace scan
}  // namespace ops_lab
