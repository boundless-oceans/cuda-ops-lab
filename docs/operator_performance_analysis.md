# 怎么看一个算子（性能分析）

> **姊妹篇**：`operator_code_navigation.md` 讲"**这个算子由哪些文件组成、该打开哪个**"。
> 这份讲"**这个算子快不快、瓶颈在哪**"。
>
> 拿到一个 kernel —— 自己写的、别人写的、或者比赛参考实现 —— 在动手改之前，
> 先用一套**固定顺序**回答三个问题：它现在快不快？瓶颈在哪？下一刀砍哪？
>
> **核心原则：先算，再读代码，最后才看工具。** 顺序反了就会陷入"到处看指标，
> 但不知道哪个重要"。

本文里所有命令都可以直接在本仓库跑。需要 GPU 的部分会明确标注。

> 下面命令里的环境名用 `opslab` —— 这是仓库 `environment.yml` 里定义的名字。
> 如果你本地用的是别的名字，自行替换。

---

## 第 0 步：先算，不读代码

三个数字，全部可以在**没读代码之前**算出来：

| 符号 | 含义 |
|---|---|
| `B` | 这次运算总共搬运多少字节（**读 + 写都要算**） |
| `F` | 总共做多少次浮点运算 |
| `I = F / B` | 算术强度，单位 FLOP/Byte |

然后跟这台机器的 **machine balance** 比。实测值用：

```bash
conda run -n opslab python scripts/check_env.py     # 看 [5/7] GPU 与性能上限
```

> `check_env.py` 的 `[5/7]` 一节会给出实测带宽、FP32 算力和 machine balance。

### 例：`add`（`out = x + y`）

| 项 | 值 |
|---|---|
| 搬运 | 读 x（4N 字节）+ 读 y（4N）+ 写 out（4N）= **12N 字节** |
| 运算 | N 次加法 = **N FLOP** |
| 算术强度 | `N / 12N` = **0.083 FLOP/Byte** |

和 machine balance（消费级 GPU 通常 30~60 FLOP/Byte）一比就清楚了：
0.083 差了 **几百倍** —— 这个算子**永远不可能**是算力瓶颈。

### 由此得到"理论最短耗时"

```
理论最短耗时 = B / 峰值带宽
```

N = 2²⁴ ≈ 16.7M 时：`12 × 2²⁴ ≈ 201 MB ÷ 256 GB/s ≈ 786 µs`

**这个数字是你唯一的判据。** 实测 800 µs 说明已经到顶了，别再优化；
实测 3000 µs 说明还有 4 倍空间。

---

## 第 1 步：判断类型 → 这一步决定后面 90% 的判断标准

| 类型 | 条件 | 优化方向 | 什么时候该停 |
|---|---|---|---|
| **memory-bound** | `I ≪ machine balance` | 一切为了**打满带宽** | 到峰值 85% 就停 |
| **compute-bound** | `I ≫ machine balance` | 一切为了**减少运算 / 用更快的指令** | 到算力 85% 就停 |

**最常见的错误**：对一个 memory-bound 算子做"减少计算量"的优化。
算术强度 0.083 的算子，你把计算量砍掉一半，耗时几乎不变 —— 因为时间花在等内存上。

---

## 第 2 步：读访存（memory-bound 算子的主战场）

按重要性排序，**前一条不过关，后面三条没意义**：

### ① 是否合并（coalesced）？

**判断方法**：取一个 warp 的 32 个连续线程，把它们访问的地址算出来，看跨度。

具体做法是把线程索引表达式代进去，算相邻两个线程的地址差：

```cuda
i = blockIdx.x * blockDim.x + threadIdx.x;   // 地址差 = 4 字节  → ✅ 完美合并
i = threadIdx.x * gridDim.x + blockIdx.x;    // 地址差 = gridDim.x × 4 → ❌ 灾难
```

第二条为什么是灾难：一个 warp 的 32 个线程会落在 **32 条不同的 cache line** 上，
一次访存请求被拆成 32 个事务。这就是"带宽只有峰值的 10~20%"的典型原因。

> 本仓库的 `v0_uncoalesced` 变体就是故意写成第二条的样子，用来做反面教材。

### ② 是否向量化？

一次访存搬 4 字节还是 16 字节（`float4`）？向量化把**访存指令数降 4 倍**，
在 memory-bound 场景下通常能再拿 10~20% 带宽。

### ③ 对齐？

`float4` 要求指针 16 字节对齐。不对齐时要么走标量路径，要么直接崩。
**正确做法是回退到标量路径，而不是报错** —— 否则合法输入会挂。

### ④ 尾巴？

`N` 不是向量宽度的整数倍时怎么办？这一段代码最容易出 bug，也是比赛里
最常见的失分点。

---

## 第 3 步：读并发度

合并访存做到了，带宽却还上不去，问题通常在这里。

| 看什么 | 在哪看 | 判断 |
|---|---|---|
| grid / block 大小 | 代码里的 `<<<grid, block>>>` | block 是不是 128/256 的倍数？grid 是不是太小？ |
| 每线程处理几个元素 | 代码 | 1 个 → 在飞访存请求少；4 个 → ILP 更高 |
| occupancy | `occupancy` API 或下面第 4 步 | 见下方说明 |
| 每线程的在飞请求数 | ncu 的 `long scoreboard` | 高 = 在等内存 |

### 关于 occupancy 的一个重要反直觉

**occupancy 低不等于性能差。**

纯流式访存的 kernel（比如 elementwise、copy），在 occupancy 只有 30% 时
**依然能打满带宽**。原因是 Little's law：

```
需要的在飞请求数 = 访存延迟 × 带宽
```

只要每个线程有足够多的独立访存请求在飞，线程数少一点没关系。
真正致命的是"**既没有线程数、也没有每线程并行度**"。

> 所以看到 occupancy 低先别急着优化，**先看带宽到没到**。
> 本仓库第 1 章的 `bandwidth_probe` 就是专门用来验证这件事的实验。

---

## 第 4 步：读寄存器压力

```bash
# 注意 --clean：ninja 是增量的，已经构建过的文件不会重新编译，
# 也就不会产生 ptxas 输出（那条 grep 会是空的）。想看资源报告得强制重编。
conda run -n opslab python python/ops_lab/_build.py --clean -v 2>&1 \
  | grep -E "registers|spill"
```

输出形如：

```
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
Used 12 registers, 384 bytes cmem[0]
```

| 看什么 | 正常 | 危险 |
|---|---|---|
| `spill stores/loads` | `0 bytes` | 非 0：寄存器不够，变量被塞进 local memory（实际上是显存），每次访问都是几百周期 |
| `Used N registers` | 越多，occupancy 上限越低 | 和 block size 一起影响能驻留几个 block |

**寄存器数是 occupancy 的隐形天花板。** 例如每线程 64 个寄存器时，
一个 SM 最多只能同时住 1024 个线程（65536 寄存器 / 64）。

---

## 第 5 步：读共享内存与同步（如果用了）

| 问题 | 症状 | 怎么查 |
|---|---|---|
| **bank conflict** | 加了共享内存反而更慢 | 看访问模式：同一个 warp 里有没有两个线程访问同一 bank 的不同地址 |
| padding / swizzle | 矩阵转置的经典解法 | 把 `smem[tid]` 改成 `smem[tid + tid/32]` 之类 |
| `__syncthreads()` 太多 | 同步开销吃掉收益 | 数代码里有几条，能不能合并 |

### 静态验证共享内存到底用没用

`cuobjdump` 的**资源报告看不到动态共享内存**（会显示 `SHARED:0`），
但 SASS 能看到真相：

```bash
cuobjdump -sass build/ext/obj/kernels/01_execution/bandwidth_probe.cu.o | grep -E "STS|LDS|BAR"
```

- `STS` = store shared，`LDS` = load shared，`BAR.SYNC` = 线程同步
- 如果代码里写了 `extern __shared__` 却搜不到 `STS/LDS`，说明**那段共享内存
  被编译器优化掉了** —— 你申请的 48KB 实际是 0，实验会静默失效

> **记住这条：资源报告里"没看到"不等于"没使用"。**

---

## 第 6 步：读边界与正确性

跑通不代表正确。按这个顺序查：

| 情况 | 期望行为 |
|---|---|
| `N == 0` | 返回空结果，**不启动 kernel** |
| `N` 小于一个 block | 至少要启动 1 个 block，否则尾巴无人处理 |
| `N` 不是向量宽度的倍数 | 标量尾巴路径 |
| 输入非连续 | 先 `contiguous()`，否则 `data_ptr()` 拿到的是错的内存 |
| 超大 `N` | grid 溢出 `2³¹-1` 时报**清晰的错**，而不是静默出错 |

有 GPU 时用官方工具兜底：

```bash
compute-sanitizer --tool memcheck python your_script.py
```

---

## 症状 → 病因 对照表（最实用的一节）

| 症状 | 最可能的原因 | 下一步查什么 |
|---|---|---|
| 带宽只有峰值 10~20% | 访存未合并 | 算相邻线程地址差；看一个 warp 跨几条 cache line |
| 带宽 60~75%，上不去 | 并发度不足（每线程在飞请求少） | 每线程处理几个元素？试 unroll 或更大 grid |
| occupancy 高但带宽低 | 访存模式差 | 回到第 2 步，别在 occupancy 上浪费时间 |
| occupancy 低但带宽高 | **正常**（Little's law） | 不用优化，去干别的 |
| 出现 local memory 读写 | 寄存器 spill | ptxas 的 spill bytes；精简变量或降 unroll |
| 加了共享内存反而更慢 | bank conflict 或同步开销 | 看访问模式；数 `__syncthreads()` 条数 |
| 实测远慢于理论下限，且不是访存问题 | 启动开销 / N 太小 / SFU 瓶颈 | 换个更大的 N 试；算一下超越函数吞吐 |
| 结果偶尔错、跑几次不一样 | 竞态 / 缺同步 / 越界 | `compute-sanitizer` |
| 编译报 `nvcc fatal: Unknown option` | nvcc 不认 gcc 的选项 | 宿主编译器选项走 `-Xcompiler`，链接器走 `-Xlinker` |

---

## 完整走一遍：本仓库的 `elementwise_add_v1_naive`

> 同一个算子，`operator_code_navigation.md` §4 从**文件组织**的角度也讲了一遍
> （哪个文件干什么）。两份对着看，能把"代码在哪"和"它快不快"连起来。

```bash
# 第 0 步：算
#   B = 12N，F = N，I = 0.083 FLOP/Byte  →  memory-bound

# 第 1 步：判断类型 → 打满带宽

# 第 2 步：读访存
#   i = blockIdx.x * blockDim.x + threadIdx.x
#   相邻线程地址差 = 4 字节 → 合并 ✅
#   未向量化 ❌（一次 4 字节）
#   无对齐要求（标量）✅
#   有边界检查处理尾巴 ✅

# 第 3 步：读并发度
#   每线程 1 个元素 → 在飞请求少
#   grid = ceil(N / 256)，覆盖满机器

# 第 4 步：读寄存器
conda run -n opslab python python/ops_lab/_build.py -v 2>&1 | grep -E "registers|spill"
#   → 12 registers, 0 spill ✅（occupancy 不会被寄存器限制）

# 第 5 步：没用共享内存 → 跳过

# 第 6 步：边界 → N=0 / 非连续 / 超大 N 都要能正确处理
```

**结论**：合并访存已达成，寄存器无压力。剩下的空间是
**向量化**（→ `v4_vectorized`）和 **提高每线程并行度**（→ `v2_unroll4`）。
这正是第 2 章阶梯的由来。

---

## 工具速查

**分析时用什么命令**（仓库里每个脚本本身是干什么的，归 `operator_code_navigation.md` 讲）：

| 命令 | 用途 | 需要 GPU |
|---|---|---|
| `python/ops_lab/_build.py --clean -v` | ptxas 资源报告（寄存器 / spill）。**必须加 `--clean`**，否则增量构建不重编、没有输出 | 否 |
| `python scripts/check_env.py` | 峰值带宽、算力、machine balance | 否 |
| `cuobjdump -sass` | 看真实生成的指令（验证共享内存、向量化） | 否 |
| `cuobjdump -res-usage` | 静态资源（注意：**看不到动态共享内存**） | 否 |
| `ncu --set full ./prog` | 访存事务数、stall 原因、occupancy 实况 | **是** |
| `compute-sanitizer` | 越界、竞态、未初始化内存 | **是** |

---

## 一句话总结

**先算算术强度定类型，再按"合并 → 向量化 → 并发度 → 寄存器 → 共享内存 → 边界"
逐层排查，每一步都对照理论下限判断该不该继续。到 85% 峰值就停手，
写清楚为什么到不了 100%。**
