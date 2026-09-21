// kernels/03_reduction/reduction_block.cuh
//
// 块内归约的三种写法 —— 这一章真正要比的三个东西。
//
// v0 只改**读**的索引公式，v4 只改**用多少 block**；v1 / v2 / v3 的差别就是
// 下面这三个函数里调哪一个。所以把块内归约抽到这里，每个变体的 .cu 只剩
// "读 → 调哪个函数 → 原子合并"三步，相邻两个变体的 diff 就是那一行调用。
//
// ## 三个函数共用的契约
//
//   * 入口 acc 是本线程的**私有累加值**；没分到元素的线程必须传 Op::kIdentity
//   * 出口**只有 threadIdx.x == 0 的返回值有意义**，调用方负责在 tid 0 上做
//     一次 `Op::atomic_combine(out, ...)`
//   * 都假设 blockDim.x == kBlockSize
//
// ## 为什么返回值只对 tid 0 有效
//
// 因为只有 tid 0 会去做原子合并。让所有线程都拿到块结果需要多一次
// `__syncthreads()`，而这一章的结论恰恰是"块内这些开销全都测不出来" ——
// 那就不该为了对称性白付一次同步。
//
// 但有个坑必须避开：v1 / v2 的结果在共享内存里，**不能**让 255 个线程去读
// 一块自己没写过的 smem（那是数据竞争，compute-sanitizer 会报）。
// 所以那里的读取写成 `threadIdx.x == 0 ? smem[0] : Op::kIdentity`，
// 谓词化的 load 只会被 tid 0 真正执行。
#pragma once

#include "kernels/03_reduction/reduction_config.cuh"
#include "kernels/03_reduction/reduction_ops.cuh"

namespace ops_lab {
namespace reduction {

// --------------------------------------------------------------- v1：smem 树
//
// 顺序寻址（sequential addressing）：第 s 级里 thread t 合并 smem[t] 与 smem[t+s]，
// s 从 blockDim/2 逐级折半。
//
// 为什么这个寻址方式本身就没有 bank conflict：每一级只让 tid < s 的线程干活，
// 同一个指令里的两张表（smem[tid] 与 smem[tid+s]）在 warp 内落到互不相同的
// bank 上，两条 load 是两个独立指令，各自都没有冲突。
//
// **这句话是 v2 存在的全部理由** —— 经典建议"归约要给 smem 加 padding 消
// bank conflict"，在顺序寻址下已经没有冲突可消了，padding 只能是个空操作。
// 但"我认为它是空操作"和"实测它是空操作"是两件事，所以 v2 还是要写、要测。
template <typename Op>
__device__ __forceinline__ float block_reduce_smem_tree(float acc) {
  __shared__ float smem[kBlockSize];

  smem[threadIdx.x] = acc;
  for (int s = kBlockSize >> 1; s > 0; s >>= 1) {
    // 同步必须在**读之前**、且必须在 if 外面：
    //   * 在 if 外面 —— __syncthreads() 要求 block 内所有线程都执行到
    //   * 在循环体开头 —— 第一次迭代要保证 smem[128..255] 已经写好了。
    //     写成循环体末尾是错的：那样第一级读 smem[tid+s] 时，
    //     tid+s 那半边的写入还没有任何屏障保护，是一个真实的数据竞争。
    __syncthreads();
    if (static_cast<int>(threadIdx.x) < s) {
      smem[threadIdx.x] = Op::combine(smem[threadIdx.x], smem[threadIdx.x + s]);
    }
  }
  // 最后一级（s == 1）就是 tid 0 自己写的 smem[0]，所以它不需要额外同步
  return threadIdx.x == 0 ? smem[0] : Op::kIdentity;
}

// ------------------------------------------------------ v2：带 padding 的 smem 树
//
// 布局：按 warp 把 kBlockSize 个值切开，**每个 warp 的 32 个值后面补 1 个空位**，
// 于是 warp w 的 lane l 落在 smem[w*33 + l]。这是"用 padding 消除 bank conflict"
// 的标准做法：原本如果两个线程访问的下标相差 32 的倍数，它们会撞同一个 bank；
// 错开 1 位之后这个规律被打破。
//
// 本变体的预期是**什么都不会发生**（≈ v1）。理由见上面 v1 那段：顺序寻址
// 本来就没有冲突可消。padding 之后每个 warp 的 bank 映射被整体挪了一位，
// 仍然是一一对应，所以既不更快也不更慢。
//
// 若实测 padding 反而慢了百分之几，说明我把 smem 树的开销估错了 —— 那也是一个
// 有用的结论（说明地址计算那两条指令并不是完全免费的）。**先测，再解释。**
constexpr int kSmemPadPerWarp = 1;
constexpr int kPaddedSmemFloats = kBlockSize + kNumWarps * kSmemPadPerWarp;

__device__ __forceinline__ int padded_index(int tid) {
  return tid + tid / 32 * kSmemPadPerWarp;
}

template <typename Op>
__device__ __forceinline__ float block_reduce_smem_tree_padded(float acc) {
  __shared__ float smem[kPaddedSmemFloats];

  smem[padded_index(threadIdx.x)] = acc;
  for (int s = kBlockSize >> 1; s > 0; s >>= 1) {
    __syncthreads();  // 位置与理由同 v1：读之前、if 之外
    if (static_cast<int>(threadIdx.x) < s) {
      const int a = padded_index(threadIdx.x);
      const int b = padded_index(threadIdx.x + s);
      smem[a] = Op::combine(smem[a], smem[b]);
    }
  }
  return threadIdx.x == 0 ? smem[0] : Op::kIdentity;
}

// ------------------------------------------------------- v3 / v4：warp shuffle
//
// 前 5 级在 warp 内用 __shfl_down_sync 完成，**一次 __syncthreads 都不需要**；
// 8 个 warp 的部分和落到 smem、同步一次；第二级（8 个值）再用 3 次 shuffle。
// 于是共享内存访问从 8×256 次降到 8 次，同步从 8 次降到 1 次。
//
// 这里有个必须守住的规则：**lane >= kNumWarps 的线程不能提前退出**，要用
// 单位元补齐。__shfl_down_sync 要求 mask 里的 lane 全部参与，少一个就是
// 未定义行为（在老架构上表现为挂死，在 Volta 之后表现为结果不可预期）。
// 这是归约里"空线程必须贡献单位元"这条规则的第二次出现 —— 第一次是
// n 不是 block 整数倍的时候。
template <typename Op>
__device__ __forceinline__ float block_reduce_warp_shuffle(float acc) {
  __shared__ float warpsums[kNumWarps];

  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    // lane + offset >= 32 时，shuffle 返回线程自己的值。这些值会被丢弃
    // （我们只取 lane 0 的结果），所以不需要额外保护。
    acc = Op::combine(acc, __shfl_down_sync(0xffffffffu, acc, offset));
  }

  if (lane == 0) {
    warpsums[warp] = acc;
  }
  __syncthreads();

  if (warp == 0) {
    float v = (lane < kNumWarps) ? warpsums[lane] : Op::kIdentity;
#pragma unroll
    for (int offset = kNumWarps >> 1; offset > 0; offset >>= 1) {
      v = Op::combine(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    acc = v;
  }
  // warp 0 的 lane 0 拿到最终结果；其余线程的返回值无意义（契约里已说明）
  return acc;
}

}  // namespace reduction
}  // namespace ops_lab
