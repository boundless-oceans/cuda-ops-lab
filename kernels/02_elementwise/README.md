# 02 Elementwise（逐元素运算）

> 本章是整条阶梯最完整的一章：三个算子 × 五个变体，而且**全部数字都是实测的**。

---

## 1. 问题定义

三个算子，输入输出同形状，逐元素独立：

| 算子 | 数学定义 | 输入 → 输出 |
|---|---|---|
| `add` | `out = x + y` | 两个 `float32[N]` → `float32[N]` |
| `relu` | `out = max(x, 0)` | `float32[N]` → `float32[N]` |
| `sigmoid` | `out = 1 / (1 + e^{-x})` | `float32[N]` → `float32[N]` |

约束：`float32`；任意 `N ≥ 0`（含 0、1、非 2 的幂、非 4 的倍数）；
输出与输入形状一致；参考实现在 `float64` 上计算。

---

## 2. 先算理论下限（★ 写代码之前）

|  | `add` | `relu` / `sigmoid` |
|---|---|---|
| 搬运 `B` | 读 x (4N) + 读 y (4N) + 写 out (4N) = **12N** | 读 x (4N) + 写 out (4N) = **8N** |
| 运算 `F` | N | N（`sigmoid` 的 `exp` 不计，见下） |
| 算术强度 `I` | **0.0833** FLOP/Byte | **0.1250** FLOP/Byte |

本机 machine balance ≈ 54 FLOP/Byte（见 `scripts/check_env.py` 的 `[5/7]` 一节）。
算术强度差了**几百倍** → **两者都是 memory-bound**，优化目标只有一个：**打满带宽**。

**理论最短耗时**（N = 2²⁴ = 16,777,216，峰值带宽实测 256.0 GB/s）：

| 算子 | 搬运量 | 理论最短 |
|---|---|---|
| `add` | 201.3 MB | **786.3 µs** |
| `relu` / `sigmoid` | 134.2 MB | **524.2 µs** |

`exp` 为什么不计入 FLOP：它既不是一个 FMA，也不是本章的知识重点。这一列只用来判断
"算术强度 vs machine balance"。**实测算不算得进都不影响结论**，见第 3 段。

---

## 3. 优化阶梯表（实测）

设备：RTX 4060 Laptop（sm_89，24 SM），峰值带宽 256.0 GB/s，N = 2²⁴。
计时：warmup 20 + 100 次迭代，每次单独 `record/sync`，取 **median**。

### `add`（201.3 MB，理论下限 786.3 µs）

| 变体 | 思路 | median (µs) | GB/s | %峰值 | 相比上一级 |
|---|---|---|---|---|---|
| `v0_uncoalesced` | 反面教材：相邻线程地址相隔 `gridDim.x` | 3685.4 | 54.6 | **21.3%** | — |
| **`v1_naive`** | 一线程一元素、已合并访存 | **839.6** | **239.8** | **93.7%** | **4.39×** |
| `v2_unroll4` | 每线程 4 元素，先全 load 再统一 store | 852.4 | 236.2 | 92.3% | 0.99× |
| `v3_grid_stride` | grid 固定成填满机器 + 循环步进 | 853.0 | 236.0 | 92.2% | 1.00× |
| `v4_vectorized` | `float4`，访存指令数降 4 倍 | 859.0 | 234.4 | 91.5% | 0.99× |
| `torch`（基线） | `x + y` | 851.9 | 236.3 | 92.3% | 最快变体是它的 0.99× |

### `relu` / `sigmoid`（134.2 MB，理论下限 524.2 µs）

| 变体 | relu GB/s | relu %峰值 | sigmoid GB/s | sigmoid %峰值 |
|---|---|---|---|---|
| `v0_uncoalesced` | 56.6 | 22.1% | 55.6 | 21.7% |
| `v1_naive` | 228.3 | 89.2% | 228.1 | 89.1% |
| `v2_unroll4` | 228.0 | 89.0% | 226.7 | 88.6% |
| `v3_grid_stride` | **230.2** | **89.9%** | 224.5 | 87.7% |
| `v4_vectorized` | 226.6 | 88.5% | 226.4 | 88.4% |
| `torch` | 227.9 | 89.0% | 228.0 | 89.0% |

> 复现：`python bench/run_bench.py --chapter elementwise`
> （原始输出在 `bench/results/02_elementwise.md`，该目录已 gitignore）

### 这张表推翻了我的预测，而且推翻得很彻底

设计文档预测的是一条爬升：`v0` 10~20% → `v1` 60~75% → … → `v4` 85~95%。

**实测是：一个大台阶，然后一条平线。**

1. **`v1_naive` 已经到 93.7% 峰值**，`v2`/`v3`/`v4`/`torch` 全在它下面 1~3%。
2. **`v4_vectorized` 反而最慢**（91.5%，三个算子上都稳定地比 `v1` 慢 1~3%）。它多出来的标量尾部代码要付代价，而"访存指令数降 4 倍"省下的是**指令**，不是**时间**。
3. **`sigmoid` 与 `relu` 的带宽一模一样**（228.1 vs 228.3 GB/s）——`exp` 那几十条指令被访存延迟**完全盖住**。这直接证实了第 2 段的判断：加上 `exp` 之后算术强度上升了，但它**仍然**是 memory-bound。

**结论：在这台机器上，`v1_naive` 就是答案。** 后四级不是"没写好"，是真的没有可优化的空间了——带宽已经打满，继续微调是在浪费生命。

---

## 4. 每级实现要点

### `v0_uncoalesced` —— 反面教材

```cuda
i = threadIdx.x * gridDim.x + blockIdx.x;   // ← 与 v1 只差这一行
if (i < n) out[i] = x[i] + y[i];
```

相邻线程的地址相隔 `gridDim.x` 个元素，一个 warp 的 32 个线程落在 **32 条不同的
cache line** 上，一次访存请求被拆成 32 个事务。实测 **21.3%** 峰值。

**但真正值得记住的是：它只慢 4.39 倍，不是教科书的 32 倍。**

教科书的"事务放大 32 倍 → 带宽掉到 1/32"**默认了没有缓存复用**。而这里
`i = tid * gridDim.x + bid` 的写法有个性质：固定 `tid`、连续 `bid` 访问的是**连续地址**。
于是同一条 cache line 会被 32 个**不同的 block** 先后用到，**L2 把它们接住了**。

所以 v0 **并没有浪费显存带宽**（逻辑字节数一样），它浪费的是 **L2 的事务吞吐**
——一个 warp 的请求从 4 个 sector 请求变成 32 个。**瓶颈从 DRAM 带宽移到了 L2 事务率。**

> 比赛里的实际意义：看到访存慢，**不要条件反射地说"没合并"**。
> 先分清是 DRAM 带宽不够，还是 L2/事务吞吐不够——两者的优化方向完全不同。
>
> 这个推断是可证伪的，验证方法见第 5 段。

### `v1_naive` —— 基准点，也是最终答案

```cuda
i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) out[i] = x[i] + y[i];
```

全局线程 id 直接映射一个元素：相邻线程访问相邻地址，**天然合并**。没有循环，
所以 grid 有 2³¹−1 的上限（超出会抛清晰错误，不是静默出错）。

它已经到 **93.7%** 峰值，比 `torch` 的 `x + y` 还快一点。

### `v2_unroll4` —— 用 ILP 换并发度

每线程 4 个元素，块内跨步：线程 `t` 拿走 `t`、`t+B`、`t+2B`、`t+3B`。关键是
**先把 4 个 x 和 4 个 y 全部 load 进来，再统一计算、统一 store**：

```cuda
if (base + stride * (kUnroll - 1) < n) {      // 快路径：4 个下标都合法
  float a[4], b[4];
  for (k) a[k] = x[base + k*stride];          // 8 个独立 load 排在一起发射
  for (k) b[k] = y[base + k*stride];
  for (k) out[base + k*stride] = a[k] + b[k];
  return;
}
for (k) { if (i < n) out[i] = x[i] + y[i]; }  // 慢路径：只有最后一个 block 会走
```

如果把 `if (i < n)` 塞进循环里，编译器无法确定 4 个 load 是否都会执行，只能一个一个来
——ILP 就没了。所以拆成快慢两条路径。

`SASS` 确认快路径确实是 **8 个连续 `LDG` 后接 4 个 `STG`**。代价是寄存器从 12 涨到 25。

**但没有换来带宽**（92.3% vs v1 的 93.7%）。原因见第 5 段的 occupancy 分析。

### `v3_grid_stride` —— 解除 grid 上限

```cuda
const int64_t stride = gridDim.x * blockDim.x;
for (int64_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += stride)
  out[i] = x[i] + y[i];
```

grid 固定成 `SM 数 × 每 SM 能驻留的 block 数`，靠循环走完全部数据。

**确定的收益是功能性的**：`v1`/`v2`/`v4` 的 grid 都受 2³¹−1 限制，v3 没有；而且循环条件
本身就是完整的边界处理，不需要单独的尾部路径。**性能上它和 v1 没有区别**（92.2%）。

顺带一个发现：`SASS` 显示 **nvcc 自动把这个循环展开了 4 路**，所以 v3 白拿了 ILP，
代价是寄存器升到 34。

### `v4_vectorized` —— `float4`

一次访存搬 16 字节而不是 4 字节，主路径访存指令数降 4 倍（SASS 确认是
`LDG.E.128` / `STG.E.128`）。尾部不足 4 个元素走标量路径。

**四个边界都处理了**：非 4 的倍数、指针未 16 字节对齐（**退回 v1 的标量路径而不是报错**
——非对齐指针是合法输入）、`n < 4`（grid 至少 1 个 block，否则尾部无人处理）、
grid 覆盖不到尾巴（复用线程 0~2 的下标）。

**结果是三个算子上它都最慢**。省下的是指令数，不是时间——因为指令数本来就不是瓶颈。

---

## 5. ncu 观察

### 先看这个：验证第 4 段那个"L2 事务吞吐"的推断

```bash
ncu -k "regex:add_v0_kernel"       --launch-count 1 --set full ./build/native/main_02_elementwise
ncu -k "regex:add_v1_naive_kernel" --launch-count 1 --set full ./build/native/main_02_elementwise
```

对比这两个数：

| 指标 | 预测 |
|---|---|
| `dram__bytes.sum` | v0 与 v1 **接近**（v0 没浪费显存带宽） |
| `gpu__time_duration.sum` | v0 是 v1 的 **4~5 倍** |
| `lts__t_sectors.sum`（L2 事务） | v0 显著更高 |

若 DRAM 字节数相近而时间差 4 倍多 → 推断成立，瓶颈在 L2/事务吞吐。
若 DRAM 字节数也差好几倍 → 推断错了，得换成"L2 复用没接住"。

### 再看：v2/v3/v4 为什么没提升

| 指标 | 看什么 |
|---|---|
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 实测 occupancy。**预计五级都在 100%** —— 因为 block=256 时是线程数先绑死（1536/256 = 6 个 block），寄存器从 12 涨到 34 都没碰到天花板 |
| `l1tex__throughput.avg.pct_of_peak_sustained_active` | L1 吞吐。若已接近 100%，说明瓶颈在访存管道而非 DRAM |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 93.7% 应能对上 |
| `gpu__time_duration.sum` | 各变体应相差无几 |

### 各变体的静态资源（`python scripts/resource_report.py`）

| 变体 | 寄存器 | spill |
|---|---|---|
| `v0_uncoalesced` | 12 | 0 |
| `v1_naive` | 12 | 0 |
| `v2_unroll4` | 25 | 0 |
| `v3_grid_stride` | 34 | 0 |
| `v4_vectorized` | 18 | 0 |

**全部零 spill。** 而且**没有一个是寄存器受限的**：sm_89 每 SM 有 65536 个寄存器、
最多 1536 线程；block=256 时 6 个 block 是线程数定的上限，34 个寄存器下也只需要
`6×256×34 = 52224 < 65536`。

所以"v2/v3/v4 寄存器变多导致 occupancy 下降"这个解释**不成立**——它们的瓶颈
压根不在这里。**光看带宽表猜不出原因，光看寄存器表也猜不出，两张表对上才敢下结论。**

---

## 6. CUDA → Ascend C 对照

| 概念 | CUDA（本章的写法） | Ascend C | 能不能平移 |
|---|---|---|---|
| 并行粒度 | 一个线程算一个元素 | 一次搬一块进 UB，用向量指令批量算 | ❌ **思维要换**。本章"一线程一元素"的结构直接搬过去会很低效 |
| 访存 | 直接读写 global，靠 cache 自动合并 | 显式 `DataCopy` GM↔UB，**手动管理** | ❌ 没有"自动合并"这回事，搬多少由你决定 |
| 对齐 | 无强制要求（不对齐就退回标量） | **32B 对齐**，长度要够整块 | ⚠️ 尾块必须单独处理，最容易出精度 bug |
| 交织并行 | `__syncthreads` / warp 自动 | `TQue` 的 Set/Wait Flag，事件驱动 | ❌ Ascend 是流水线，不是屏障式 |
| 向量化 | `float4`（本章 v4） | 天生就是向量指令，不需要"向量化"这一步 | ✅ 但 Ascend 上这是默认行为，不是优化 |
| 双缓冲 | 手动 ping-pong | `TQue` 深度设 2 自动双缓冲 | ✅ 更"声明式" |

**给比赛的三条结论**：

1. 本章阶梯在 Ascend C 上**大部分不成立**——`v2`/`v4` 这类"提高每线程并行度 / 向量化"
   的招数，Ascend C 里要么是默认行为，要么根本不是那个模型。
2. `v0 → v1` 那一级**有意义**：它说的是"访存要连续"。Ascend C 里对应"`DataCopy`
   搬的块要连续"，错了同样代价惨重。
3. **真正要学的是第 4 段那个教训**：先分清瓶颈是在 DRAM 带宽还是在事务/搬运吞吐。
   Ascend 上这个区分同样存在（搬运指令数 vs 搬运字节数）。

---

## 7. 踩坑记录

| 现象 | 真相 |
|---|---|
| `nvcc fatal : Unknown option '-fPIC'` | nvcc 不认，要 `-Xcompiler=-fPIC` |
| `nvcc fatal : Unknown option '-Wl,-rpath,...'` | 要 `-Xlinker -rpath -Xlinker <dir>`；且**不能**写 `-Xcompiler=-Wl,-rpath,X`（该值逗号分隔，会被拆成三个选项而静默失效） |
| `-Wrestrict`: argument 2 aliases with argument 4 | v4 里把 `out4` 和 `out` 同时作为两个 `__restrict__` 参数传进 kernel，而它们指向**同一块内存** —— 违反 restrict 承诺、属未定义行为。改成只传 `float4*`、标量视图在 kernel 内派生 |
| `cuobjdump -res-usage` 显示 `SHARED:0` | **动态共享内存不出现在静态资源报告里**，那是启动参数。要看真实用量得看 SASS 里有没有 `STS`/`LDS` |
| 测试里 `hello/v2_grid_stride` 报 TypeError | 元数据声明 1 个参数、绑定要 3 个。**离线就能查**，只是当时漏了——见 `tests/test_metadata.py` 的 arity 检查 |
| 基准数字虚高 | 测试形状最大的工作集只有 12 MB，装得进 L2。所以基准要用独立的 `bench_shapes`（2²⁴，201 MB） |
