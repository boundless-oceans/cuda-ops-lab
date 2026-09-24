# 03 Reduction（归约）

> 这一章的核心结论很反直觉：**块内归约怎么优化都不影响时间**。
> SASS 能证明优化真的生效了（同步 8 次 → 1 次，共享内存访问 26 次 → 2 次），
> 而耗时一动不动 —— 因为按理论下限算，整个块内归约只占 **0.1%**。
> 真正贵的是另一件事：**用了几个 block**（v4 只用一个，慢 16.3 倍）。

---

## 1. 问题定义

| 算子 | 数学定义 | 输入 → 输出 | `bytes` | `flops` |
|---|---|---|---|---|
| `sum` | `y = Σ x[i]` | `float32[N]` → `float32[1]` | 4N + 4 | N − 1 |
| `max` | `y = max(x[i])` | `float32[N]` → `float32[1]` | 4N + 4 | N − 1 |

约束：`float32`；任意 `N ≥ 0`（含 0、1、非 2 的幂）；**输出形状恒为 `(1,)`**；
参考实现在 `float64` 上计算。

**和第 2 章的根本区别有两条**，它们决定了这一章的全部内容：

1. **输出比输入小 4N 倍** —— 搬运量几乎是纯读（`4N + 4`）。所以"带宽"这个词要小心：
   真正搬的是读的那一份，写的 4 个字节可以忽略。
2. **元素之间产生了依赖** —— N 个值必须合并成 1 个，于是绕不开
   "线程内串行累加 → 块内合并 → block 之间汇总" 这三层结构。
   第 2 章每个元素独立，永远不需要 `__syncthreads()`；这里必须有。

**两个算子的容差不一样，是刻意的**（见 `python/ops_lab/reduction.py`）：

| 算子 | 容差 | 理由 |
|---|---|---|
| `max` | **`rtol = atol = 0`（精确相等）** | 只有比较、没有算术，结果是精确的。比给一个 1e-5 的容差强得多 |
| `sum` | `rtol = 1e-4` | 浮点累加，误差随累加深度增长（N ≈ 1e6 时最坏约 1e-4 相对） |

`sum` 的测试输入故意用**全正数**（`rand` 而不是 `randn`）：大 N 下 `randn` 会正负相消，
相对误差失去意义，容差就变成玄学了。而 `max` 的输入故意**带负数**（`randn`），
专门用来测**单位元** —— 如果 `max` 的单位元错写成 `0.0f`，全正数输入下一点问题都没有。

---

## 2. 先算理论下限（★ 写代码之前）

N = 2²⁴ = 16,777,216（与第 2 章同形状，便于横向比较）：

| 项 | 值 |
|---|---|
| 搬运 `B` | 读 4N + 写 4 = **67.1 MB** |
| 运算 `F` | N − 1 ≈ 1.68e7 FLOP |
| 算术强度 `I` | **0.25 FLOP/Byte**（第 2 章的 `add` 是 0.083） |
| 判定 | machine balance ≈ 54 FLOP/Byte → **memory-bound，差 200 多倍** |
| **理论最短耗时** | 67.1 MB ÷ 256.0 GB/s = **262.1 µs** |

### ★ 但这一章真正要算的是另一笔账

能不能提前知道"块内归约的优化值不值钱"？可以，只要估一下它的量级：

```
块内树一共 8 级（256 → 128 → … → 1）
每级 = 1 次 __syncthreads + 1 次 smem 读写 ≈ 50 cycle
8 级 ≈ 400 cycle ≈ 0.25 µs（sm_89 约 1.6e9 cycle/s）
```

**0.25 µs ÷ 262.1 µs = 0.1%。** 也就是说：把 8 级压成 1 级、把同步全删掉，
最多也就能省下 0.1% —— 比测量噪声还小。

所以本章在写代码之前就下了一个**可证伪的预测**：

> 网上流传的经典归约阶梯（2007 年那份 *Optimizing Parallel Reduction in CUDA*，
> 5 级爬升、号称 5 倍加速）**在这台机器上不应该重现**。
> v1 / v2 / v3 应当落在一条平线上。

这个预测**成立**了，而且是这一章最有价值的结论 —— 它同时教了两件事：
经典阶梯在当年是对的（那时块内归约占比高、全局访存远没打满），
以及**知道"什么不值得优化"和知道"什么值得优化"同样重要**。

---

## 3. 优化阶梯表（实测）

设备：RTX 4060 Laptop（sm_89，24 SM），N = 2²⁴，理论下限 262.1 µs。

**测时环境**（这两个数字是这张表能不能信的前提）：

```
预热：显存时钟 8001 / 8001 MHz（满速） · SM 2445 MHz · 46°C · 55W · 预热 1.6s
测时显存时钟 8001 / 8001 MHz（满速）
计时：3 轮交错（轮间起始变体错开），每轮 warmup 20 + 40 次独立 record/sync
```

> 为什么要专门写时钟：这台机器冷态显存时钟只有 7001 MHz（额定的 87.5%），
> 而 `%峰值` 的分母用的是额定的 8001 MHz。**冷机开测，最先测的几个变体会被凭空
> 打上"只有 83% 峰值"的标签。** 这个坑在第 3 章踩过一次（两条测量路径差 14%），
> 完整还原见 [`docs/operator_performance_analysis.md`](../../docs/operator_performance_analysis.md)
> 的"测之前"一节。

### `sum`（67.1 MB）

| 变体 | 思路 | median (µs) | 轮间极差 | GB/s | %峰值 | 相比上一级 |
|---|---|---|---|---|---|---|
| `v0_uncoalesced` | 反面教材：相邻线程地址相隔 `gridDim.x` | 578.5 | 6.3 | 116.0 | **45.3%** | — |
| **`v1_naive`** | grid-stride 合并访存 + smem 顺序寻址树 + 每 block 一次原子 | **276.4** | 1.0 | 242.8 | **94.8%** | **2.09×** |
| `v2_smem_tree` | smem 加 padding（`tid` → `tid + tid/32`） | 275.5 | 1.6 | 243.6 | 95.2% | 1.00× |
| **`v3_warp_shuffle`** | 块内改用 `__shfl_down_sync` | **274.6** | 1.0 | **244.4** | **95.5%** | 1.00× |
| `v4_single_block` | 故意回退：块内沿用 v3，但 `grid = 1` | 4516.9 | 0.1 | 14.9 | 5.8% | 0.06× |
| `torch`（基线） | `x.sum()` | 277.5 | 1.0 | 241.8 | 94.5% | 最快变体是它的 0.99× |

### `max`（67.1 MB）

| 变体 | 思路 | median (µs) | 轮间极差 | GB/s | %峰值 | 相比上一级 |
|---|---|---|---|---|---|---|
| `v0_uncoalesced` | 反面教材：访存未合并 | 572.5 | 2.4 | 117.2 | **45.8%** | — |
| `v1_naive` | 同 `sum`，合并用 `fmaxf`、汇总用 CAS 版 `atomicMax` | 276.5 | 0.5 | 242.7 | 94.8% | **2.07×** |
| `v2_smem_tree` | smem 加 padding | 275.5 | 1.2 | 243.6 | 95.2% | 1.00× |
| **`v3_warp_shuffle`** | 块内改用 `__shfl_down_sync` | **275.4** | 0.1 | 243.6 | 95.2% | 1.00× |
| `v4_single_block` | `grid = 1` | 4516.9 | 0.0 | 14.9 | 5.8% | 0.06× |
| `torch`（基线） | `x.max()` | 276.5 | 2.0 | 242.7 | 94.8% | 最快变体是它的 1.00× |

> 复现：`python bench/run_bench.py --chapters reduction`
> （原始输出在 `bench/results/03_reduction.md`，该目录已 gitignore）

### 这张表推翻了我的一个预测，但保住了最重要的那个

**我预测错的**：v0 我写的是"~20%（慢 4~5 倍）"，实测 **45.3% / 45.8%（只慢 2.09 倍 / 2.07 倍）**。

和第 2 章是同一个教训，而且这次更极端 —— 第 2 章那个写法的代价是 4.35 倍，
归约里只有 2.1 倍。原因还是 **L2 把复用接住了**：索引写成
`i = tid * gridDim.x + bid` 之后，**固定 `tid`、连续 `bid` 访问的是连续地址**，
所以同一条 cache line 会被 32 个不同的 block 先后用到。
v0 没有浪费显存带宽，它浪费的是 **L2 的事务吞吐**。

教科书的"事务放大 32 倍 → 带宽掉到 1/32"**默认了没有缓存复用**。
实测 2.1 倍，比 32 倍少了整整一个数量级。

**我预测对的（而且是本章的重点）**：

1. **v1 / v2 / v3 是一条平线**：`v0` 之后是 **94.8% → 95.2% → 95.5%**，
   相邻两级的"相比上一级"全是 **1.00×**。三级之间最大差 1.8 µs（0.65%）——
   比"经典阶梯"号称的 5 倍少了将近三个数量级。
   注意这 0.65% **不能一口咬定是噪声**：它略大于各自的轮间极差（1.0 µs），
   所以可能是 shuffle 带来的真实小幅收益。但方向改变不了结论：
   **量级上这就是平线。**
2. **v4 只用一个 block，慢 16.3 倍**（4516.9 µs vs 276.4 µs）。
   注意 v4 的块内写法是**本章最激进的那一套**（v3 的 shuffle），
   它证明：**块内优化再精致，也救不回"只用 1/24 的机器"**。
3. **`v1_naive` 就是答案**：94.8% 已到顶，比 `torch` 还快 0.4%。
   后面两级（v2/v3）不是"没写好"，是真的没有可优化的空间。

### 一个横向对比（三章的峰值利用率放在一起）

| 章节 | 访存形态 | 最好成绩 | %峰值 |
|---|---|---|---|
| 第 1 章 `hello` | 纯写 4N | 253.0 GB/s | **98.8%** |
| 第 3 章 `sum`/`max` | 纯读 4N | 244.4 GB/s | **95.5%** |
| 第 2 章 `add` | 读 2N + 写 N | 239.2 GB/s | **93.4%** |

同一条阶梯、同一台机器，**纯写 > 纯读 > 读写混合**，而且三者都 ≥93%。
一个**未经验证**的候选解释：显存总线上读写换向（bus turnaround）要付代价，
所以混合流的效率最低。这个规律值得记住 —— 它意味着
**"%峰值 上不去"未必是你的 kernel 的问题，可能只是这个访存形态的上限就在那里**。

---

## 4. 每级实现要点

### `v0_uncoalesced` —— 反面教材

```cuda
// 与 v1 只差这一行：两个乘法项换了位置
for (int64_t i = threadIdx.x * gridDim.x + blockIdx.x; i < n; i += stride)
  acc = Op::combine(acc, x[i]);
```

相邻线程的地址相隔 `gridDim.x` 个元素（本机 144 个 = 576 字节），
一个 warp 的 32 个线程落在 32 条不同的 cache line 上。
实测 **45.3% / 45.8%** —— 只慢 2.1 倍。

**仍然正确的理由**（这一条必须自己想清楚，否则会写出丢元素的版本）：
`(threadIdx.x, blockIdx.x) → threadIdx.x * gridDim.x + blockIdx.x` 是
`[0, blockDim.x * gridDim.x)` 上的**一一映射**，而循环步长恰好等于这个区间长度，
所以每个下标被恰好一个线程访问一次。

### `v1_naive` —— 基准点，也是答案

```cuda
float acc = Op::kIdentity;                       // ① 读：grid-stride，合并访存
for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
  acc = Op::combine(acc, x[i]);

smem[threadIdx.x] = acc;                         // ② 块内 smem 顺序寻址树
for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
  __syncthreads();
  if (threadIdx.x < s) smem[threadIdx.x] = Op::combine(smem[threadIdx.x], smem[threadIdx.x + s]);
}

if (threadIdx.x == 0) Op::atomic_combine(out, smem[0]);   // ③ 每 block 一次原子
```

三个值得说的决定：

**① grid 截断到 144 个 block（`SM 数 × 每 SM 块数`），靠 grid-stride 走完数组。**
不这么做的话，2²⁴ 个元素按 256 一线程要 65536 个 block，也就是 **65536 次打在同一个
地址上的原子操作** —— 它们会被 L2 串行化，代价可能比整个 kernel 还大。
截断之后只剩 144 次。代价是每个线程要走 400 多轮循环。

**② 用顺序寻址（`smem[tid]` 与 `smem[tid + s]`），不是交错寻址。**
这个写法**本身就没有 bank conflict**：每一级只让 `tid < s` 的线程干活，
同一个指令里的两张表在 warp 内落到互不相同的 bank 上。
**这句话是 v2 存在的全部理由。**

**③ `__syncthreads()` 必须在读之前、且在 `if` 外面。**
第一版我写在循环体末尾 —— 那样第一级读 `smem[tid + s]` 时没有任何屏障保护那半边的
写入，是**真实的数据竞争**（离线检查全过，只有代码审查或 racecheck 能抓）。

### `v2_smem_tree` —— 给共享内存加 padding

```cuda
smem[tid + tid / 32] = acc;    // 每个 warp 的 32 个值后面补 1 个空位
```

这是教科书里"用 padding 消 bank conflict"的标准做法。**但这里的预期是什么都不会发生**，
理由是上面 ② 那段：顺序寻址本来就没有冲突可消。

**实测确实是空操作（1.00×）。** 而 SASS 还给出了一个额外的证据：

| kernel | 总指令 | `BAR.SYNC` | `LDS` | `STS` | `SHFL` |
|---|---|---|---|---|---|
| `v1_naive<SumOp>` | 248 | 8 | 17 | 9 | 0 |
| `v2_smem_tree<SumOp>` | **288** | 8 | 17 | 9 | 0 |
| `v3_warp_shuffle<SumOp>` | 248 | **1** | **1** | **1** | **9** |

**v2 的指令数比 v1 多 40 条**（padding 要额外算地址：IMAD 71→75、IADD3 32→39）。
它做的是**更多**的活儿，却和 v1 一样快 —— 因为这点开销同样落在那个 0.1% 里。

> 这一级的价值不在"更快"，而在**验证一条经典建议的适用条件**。
> "归约要给 smem 加 padding"这句话，在**交错寻址**下是对的，在**顺序寻址**下是多余的。

### `v3_warp_shuffle` —— 块内改用 shuffle

前 5 级（256 → 8）用 `__shfl_down_sync` 在 warp 内完成，**一次 `__syncthreads` 都不需要**；
8 个 warp 的部分和落到 smem、同步 1 次；第二级（8 → 1）再用 3 次 shuffle。

**SASS 证明优化真的生效了**：`BAR.SYNC` 8 → 1，`LDS` 17 → 1，`STS` 9 → 1，
新增 9 条 `SHFL`。共享内存访问从每 block 8×256 次降到 8 次。

**而耗时一动不动**（274.6 vs 276.4 µs）。这就是本章要教的那件事：
**"优化生效了"和"优化有用"是两个独立的问题。**
先确认前者（看 SASS / 看指令数），再用理论下限判断后者值不值得做。

这里还有个必须守住的规则：**`lane >= kNumWarps` 的线程不能提前退出**，要用单位元补齐。
`__shfl_down_sync` 要求 mask 里的 lane 全部参与，少一个就是未定义行为
（Volta 之后表现为结果不可预期）。这是"空线程必须贡献单位元"这条规则的**第二次**出现 ——
第一次是 `n` 不是 block 整数倍的时候。

### `v4_single_block` —— 故意回退

```cuda
constexpr unsigned int kSingleBlockGrid = 1;   // 与 v3 的 diff 只有这一行
```

块内沿用 v3 的写法，只把启动时的 grid 从 144 改成 1。**实测慢 16.3 倍**（5.8% 峰值）。

放在阶梯最后一级是刻意的：它对比的对象是**已经调到位的 v3**，而不是从零开始的 v0。
所以表上读到的就是"块内优化全部做完之后，用错 block 数照样掉一个数量级"。

> 放到比赛里：如果你的 kernel 只有 1~2% 峰值，**先数一下到底用了几个 block**，
> 再去研究块内怎么写。顺序反了会白干很久。

---

## 5. ncu 观察

### 先看这个：v1 vs v3 —— 证明"优化生效但不重要"

```bash
ncu -k "regex:reduce_v1_naive_kernel.*SumOp"     --launch-count 1 --set full ./build/native/main_03_reduction
ncu -k "regex:reduce_v3_warp_shuffle_kernel.*SumOp" --launch-count 1 --set full ./build/native/main_03_reduction
```

> 这里刻意用 `.*SumOp` 而不是完整的 `..._kernelINS0_5SumOp`：ncu 的 `-k` 匹配的可能是
> **反混淆**后的名字（`…reduce_v1_naive_kernel<ops_lab::reduction::SumOp>(float const*, …)`），
> 写死 mangled 形式会一个都匹配不上。`.*` 两种形式都能中。
> 若仍然没有匹配，加上 `--kernel-name-base mangled` 再试。

| 指标 | 预测 |
|---|---|
| `gpu__time_duration.sum` | 两者**几乎相同**（差 <2%）—— 这是本章的结论 |
| `smsp__inst_executed_op_shared_ld.sum` | v3 应降约 **16 倍**（SASS 里 `LDS` 17→1，动态是 8×256→8） |
| `smsp__inst_executed_op_shared_st.sum` | v3 应降约 **9 倍** |
| `smsp__average_warps_issue_stalled_barrier` | v3 应**显著更低**（8 次屏障 → 1 次） |

**这组对比的意义**：如果时间相同而指令数确实降了，就同时证明了两件事 ——
优化是真的做了，而且它真的不重要。**若时间差出 5% 以上，说明我那个"0.1%"的账估错了。**

### 再看：v0 为什么只慢 2.1 倍（而不是 32 倍）

```bash
ncu -k "regex:reduce_v0_uncoalesced_kernel" --launch-count 1 --set full ./build/native/main_03_reduction
ncu -k "regex:reduce_v1_naive_kernel"       --launch-count 1 --set full ./build/native/main_03_reduction
```

| 指标 | 预测 |
|---|---|
| `dram__bytes.sum` | v0 与 v1 **接近**（v0 没有浪费显存带宽） |
| `lts__t_sectors.sum`（L2 事务） | v0 **显著更高**（一个 warp 的请求从 4 个 sector 变成 32 个） |
| `gpu__time_duration.sum` | v0 是 v1 的 **2.1 倍** |

DRAM 字节数相近而时间差 2.1 倍 → 瓶颈在 **L2 事务吞吐**，不在 DRAM 带宽。
若 DRAM 字节数也差几倍，那就得换成"L2 复用没接住"这个解释。

### 最后：确认 v4 只用了 1 个 block

```bash
ncu -k "regex:reduce_v4_single_block_kernel" --launch-count 1 --set full ./build/native/main_03_reduction
```

| 指标 | 预测 |
|---|---|
| `sm__ctas_launched.sum` | **1** |
| `gpu__time_duration.sum` | v3 的 **16.3 倍** |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 约 6% |

### 一个已知的固定开销

每次调用 launcher 都会**多启动一个 1 线程 kernel**（`init_scalar_kernel`，写单位元）。
它的耗时可以在 ncu 里直接看到（预测 1~2 µs），也是"最快变体是理论下限的 1.05×"
里那 5% 的一部分。之所以不做优化：`cudaMemsetAsync` 写不出 `-inf`，
而把初始化塞进主 kernel 需要一次全局屏障 —— 那是 CUDA 没有的便宜东西。

---

## 6. CUDA → Ascend C 对照

| 概念 | CUDA（本章的写法） | Ascend C | 能不能平移 |
|---|---|---|---|
| 归约的层次 | 线程内串行累加 → 块内树 → 原子汇总 | 向量指令 + `WholeReduce` / `BlockReduce` | ✅ 层次一样，但块内那棵树**不用你写** |
| warp shuffle | `__shfl_down_sync`（本章 v3） | **没有等价物** | ❌ Ascend 是 SIMD，一个向量指令处理 256B；"warp 内交换"这个模型不存在 |
| 共享内存 | `__shared__` + `__syncthreads()` | UB + `TQue` 的 Set/Wait Flag | ❌ Ascend 是事件驱动的流水，不是屏障式 |
| 块间汇总 | 每 block 一次 `atomicAdd` / CAS | 多核之间的累加要自己做（GM 上的原子或分级累加） | ⚠️ 概念有，但代价结构完全不同 |
| "用几个核" | grid 大小（本章 v4 的教训） | block 数 = 用几个 AI Core | ✅✅ **这条最重要**：AI Core 只有几十个，用不满就是直接按比例亏 |
| 访存对齐 | 无强制要求 | **32B 对齐**，长度要够整块 | ⚠️ 尾块要单独处理，最容易出精度 bug |

**给比赛的三条结论**：

1. **本章的块内阶梯（v2/v3）在 Ascend C 上大部分不成立。**
   padding 消 bank conflict、warp shuffle 这些招数，在 Ascend C 里要么是编译器/
   指令集已经替你做了（`BlockReduce`），要么根本不存在对应模型。
   **别把 v2/v3 当成"必须掌握的技巧"背过去。**
2. **v4 的教训是这一章最值得带走的**：AI Core 数量只有几十，
   "用满所有核"这件事的权重比 CUDA 上更高（CUDA 至少还有 24 个 SM × 多 block）。
   先确认用满了核，再谈块内优化。
3. **"先算理论下限"这一步在 Ascend C 上更关键**，因为那边的性能判据
   （搬运效率、流水占比）比 CUDA 更难凭直觉估。而本章这笔账
   （"块内只占 0.1%，所以不用管"）恰恰是把时间省下来的那个判断。

---

## 7. 踩坑记录

| 现象 | 真相 |
|---|---|
| g++ 在 `threadIdx` / `__syncthreads` / `atomicAdd` 上报一片"未声明" | 绑定层的 `.cpp` 是**宿主**编译单元，它 include 的章节头**不能**间接包含带 `__device__` 函数的头。修法：常量拆到 `reduction_config.cuh`，设备头只给 `.cu` 用（见 `docs/DESIGN.md` §5.1） |
| 块内树结果偶发错误 | `__syncthreads()` 写在了循环体**末尾** → 第一级读 `smem[tid+s]` 时没有任何屏障保护那半边的写入。必须在**读之前**（且在 `if` 外面）。**编译和元数据检查全过，只能靠代码审查或 racecheck 抓** |
| 浮点 `atomicMax` 不存在 | CUDA 只有 `atomicMax(int*)`。浮点版要用 CAS 自拼，且**比较必须在浮点域做** —— 负数的 IEEE 位模式序与整数序相反（`-0.0f` 的位模式比 `+0.0f` 大） |
| `CUDART_INF_F` 不能当 host 常量 | 它是 `__int_as_float(0x7f800000U)`，是个**设备端函数**，没法在 host 侧的常量表达式里用。单位元改用 `std::numeric_limits<float>::infinity()` |
| 全负数输入时 `max` 给出 0 | 单位元错写成 `0.0f`。**单位元是运算定义的一部分**，必须和 `combine` 写在一起（`SumOp` / `MaxOp`），换运算就同时换掉两者 |
| `n = 0` 时输出是随机数 | 主 kernel 不启动，但输出仍必须被写出来（`sum` → 0，`max` → -inf）。**把"写单位元"放在 launcher 里**，绑定层就不需要为 `n = 0` 写特例，这条路径也就不可能被漏掉 |
| 归约的测试容差不能全局统一 | `max` 只有比较、没有算术，可以要求**精确相等**；`sum` 是浮点累加，得给一个说得清来源的容差（1e-4） |
| 大 N 掩盖单元素错误 | N = 2²⁴ 时漏算一个元素只差 6e-8，比容差还小。所以"分区恰好覆盖一次"必须用**小 N 且卡在边界上**的形状查（`tests/test_03_reduction_edge.py` 用**全 1 输入**：那种输入下 float32 求和是精确的，可以要求逐位相等） |
| 同一 kernel 两条路径测出 313 vs 275 µs | **显存时钟**：冷态 7001 MHz、稳态 8001 MHz，差 14%，而 `%峰值` 的分母是额定的 8001。换算成真实利用率是 95.6% vs 95.7% —— 详见 `docs/operator_performance_analysis.md` 的"测之前"一节 |
| native 线的 `sum_v0` 比 Python 线慢 17%（691 vs 578 µs） | **未解**。数据不是原因（native 的 sum/max 用同一份数据，差 21%），顺序也不是（独立诊断脚本里它同样是第一个测的，却是 586.7 µs）。结论不受影响，但这一格如实标为未解 |

更细的方法学记录在 `docs/DESIGN.md` 的 §2.4（时钟）与 M2-2b 台账里。
