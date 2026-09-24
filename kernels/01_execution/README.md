# 01 Execution（执行模型）

> 这一章**不是优化阶梯**，而是两个受控实验。它不追求带宽数字，只负责建立后面
> 十章都要用的心智模型。

---

## 1. 问题定义

两个实验，性质完全不同：

| 实验 | 做什么 | 要回答的问题 |
|---|---|---|
| `hello` | `out[i] = i`，两个变体 | grid/block/thread 怎么映射？**grid 有上限吗？** |
| `bandwidth_probe` | 纯拷贝 `out[i] = x[i]`，四个变体 | **occupancy 掉下来，带宽会掉吗？** |

`bandwidth_probe` 的四个变体是**同一份拷贝逻辑**，唯一区别是动态共享内存占用
（0 / 24 / 32 / 48 KB）——用 smem 当"occupancy 旋钮"，不改一行 kernel 代码。

---

## 2. 先算理论下限

| 实验 | 搬运 `B` | 算术强度 | 理论最短（256.0 GB/s，N = 2²⁴） |
|---|---|---|---|
| `hello` | 4N（**只写**：`out[i] = i`） | 0.25 FLOP/Byte | **262.1 µs** |
| `bandwidth_probe` | 8N（读 4N + 写 4N） | 0 | **524.2 µs** |

两者都远低于 machine balance（≈54 FLOP/Byte）→ **都是 memory-bound**。

---

## 3. 实测阶梯表

设备：RTX 4060 Laptop（sm_89，24 SM），峰值带宽 256.0 GB/s，N = 2²⁴。

### `hello`（67.1 MB，理论下限 262.1 µs）

| 变体 | 思路 | median (µs) | GB/s | %峰值 |
|---|---|---|---|---|
| **`v1_naive`** | 一线程一元素、精确 grid、带边界检查；**grid 有上限** | **302.6** | **221.8** | **86.6%** |
| `v2_grid_stride` | grid 固定、循环步进；**无 grid 上限** | 316.4 | 212.1 | 82.8% |

### `bandwidth_probe`（134.2 MB，理论下限 524.2 µs）

| 变体 | 动态 smem | 理论 occupancy | median (µs) | GB/s | %峰值 |
|---|---|---|---|---|---|
| `probe_smem0` | 0 KB | ~100% | 593.0 | 226.3 | 88.4% |
| `probe_smem24k` | 24 KB | ~67% | 594.9 | 225.6 | 88.1% |
| `probe_smem32k` | 32 KB | ~50% | 593.9 | 226.0 | 88.3% |
| **`probe_smem48k`** | 48 KB | **~33%** | **564.2** | **237.9** | **92.9%** |

> 复现：`python bench/run_bench.py --chapters execution`
> 或纯 CUDA：`./build/native/main_01_execution`

（occupancy 那一列是理论值，`./build/native/main_01_execution` 会打印实测值。）

---

## 4. 两个实验各自说明了什么

### 实验一：`hello` —— "grid 有没有上限"比"快不快"更重要

两个变体只差 4.6%，而且**免上限的那个反而慢**（82.8% vs 86.6%）。

这和第 2 章的结论完全一致：`v2_grid_stride` 相对 `v1_naive` 也是 0.96×。所以
**"grid-stride 更快"这个流传很广的说法，在纯流式场景下不成立**——它的价值是
**功能性的**：

- `v1` 的 grid = `ceil(n / 256)`，所需 block 数超过 `gridDim.x` 上限（2³¹−1）时
  会抛清晰错误；`v2` 的循环步进天然覆盖任意大的 n
- `v2` 的循环条件本身就是完整的边界处理，不需要单独的尾部路径

顺带说明为什么慢一点：`v2` 的每线程多做一次循环判断，而且 `SASS` 显示 nvcc 会把
grid-stride 循环**自动展开 4 路**——白拿了 ILP，也白付了寄存器。

### 实验二：`bandwidth_probe` —— 预测错了一半，而且错得有意思

**原预测**：occupancy 从 ~100% 压到 ~33%，带宽**基本不变**（Little's law：每线程
多个在飞访存请求可以补偿线程数不足）。

**实测**：

- 0 / 24K / 32K 三档确实**几乎完全持平**（226.3 / 225.6 / 226.0 GB/s，差 0.3%）
  → 这部分预测**成立**
- 但 **48K 那一档反而快了 5%**（237.9 GB/s），而它正是 occupancy 最低的一档
  → 这部分预测**错了，而且方向相反**

**"占用率低反而更快"是反直觉的**，目前没有证实过的解释。一个**未经证实的**假设是：

> 每 SM 驻留的 block 越多，同时在飞的**独立访存流**就越多（6 block/SM × 24 SM
> = 144 条流，而 2 block/SM 只有 48 条）。流越多，DRAM 的行缓冲局部性越差；
> 占用率低反而让访问模式更集中。

但这解释不了为什么 4 block（67%）→ 3 block（50%）几乎没有变化（225.6 → 226.0），
却在 3 → 2 时突然跳了 5%。**趋势不单调，所以更可能是别的原因。**

**下一步该怎么查**（见第 5 段）：

1. **先确认可复现**——重跑几次，看 564 µs 是不是稳定。也值得把变体顺序反过来跑，
   排除"跑在最后的那个恰好赶上 GPU 频率/温度状态更好"这类**顺序效应**
2. 若可复现，用 ncu 看 `dram__throughput`、`lts__t_sectors`、以及 DRAM 行命中相关指标

> 这一档数字**不要写进任何结论**，直到它被复现和解释。

---

## 5. ncu 观察

### 先解决上面那个未解之谜

```bash
./build/native/main_01_execution          # 看四档 occupancy 的实测值
ncu -k "regex:copy_smem_kernel" --launch-count 1 --set full \
    ./build/native/main_01_execution 16777216 5
```

48K 与 24K 走的是**同一个 kernel**（`copy_smem_kernel`），只是启动时的 smem 参数不同
——所以两者 SASS 完全相同，差异只可能来自运行期状态。这正是它值得查的原因：
**同一份代码、同一个 kernel，只是 occupancy 不同。**

要对比的指标：

| 指标 | 看什么 |
|---|---|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 是否真的更高 |
| `dram__bytes.sum` | 搬运字节数应当一致（逻辑上都是 134.2 MB） |
| `lts__t_sectors.sum` | L2 事务数 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 实测 occupancy，验证 33% vs 67% |

### 实验一该看的

| 指标 | 看什么 |
|---|---|
| `dram__throughput...` | 两者应接近（86.6% vs 82.8%） |
| `smsp__inst_executed.sum` | v2 的循环多出指令，这能解释它为什么略慢 |

---

## 6. CUDA → Ascend C 对照

| 概念 | CUDA（本章） | Ascend C | 能不能平移 |
|---|---|---|---|
| 并行层级 | grid / block / thread，三级 | block（核）+ 核内向量流水 | ⚠️ 概念对应，但 Ascend 核数少得多（几十个），单核要干更多活 |
| 占用率 | occupancy，靠 smem/寄存器当旋钮 | 没有"occupancy"这个说法 | ❌ **本章实验二在 Ascend C 上不成立** |
| 数据搬运 | 直接读写 global，靠硬件自动合并 | 显式 `DataCopy` GM↔UB | ❌ 搬多少、搬几趟由你决定 |
| 掩盖延迟 | 靠 warp 切换 + 在飞请求 | 靠 `TQue` 双缓冲的流水 | ⚠️ 目标一样（掩盖延迟），手段完全不同 |
| 同步 | `__syncthreads()` | `TQue` 的 Set/Wait Flag | ❌ 屏障式 vs 流水线式 |

**给比赛的三条结论**：

1. **本章最值得带走的是实验二的"控制变量"方法**：同一份代码、只改一个运行期参数、
   **先写下预测**、再用实测打脸或确认。这个方法在 Ascend C 上一样适用。
2. "occupancy" 这个具体概念在 Ascend C 里没有对应物，别硬套。
3. "grid-stride 解决规模上限" 的思路在 Ascend C 里对应的是**分块循环**（tiling loop）
   ——不是因为 grid 有上限，而是因为 UB 装不下整块数据。**同一个手段，不同的理由。**

---

## 7. 踩坑记录

| 现象 | 真相 |
|---|---|
| `extern __shared__` 声明了但没被真正使用 | 编译器可能把整段动态 smem 优化掉，occupancy 实验**静默失效**（看起来申请了 48KB，实际是 0）。所以 kernel 里真的读写了一次 smem，并用 `SASS` 验证出现了 `STS`/`LDS` |
| `cuobjdump -res-usage` 显示 `SHARED:0` | **动态共享内存不出现在静态资源报告里**——它是启动参数，不是编译期属性。别用它验证这个实验 |
| native 可执行文件 `terminate called ... std::invalid_argument` | `hello_v2_grid_stride` 收到 `grid_size = 0`。"0 表示自动"这个约定当时**只在 Python 绑定层实现**，native 线不知道就直接抛了。已把约定挪进 launcher，两边统一 |
| 基准报告写"搬运 67.1 MB（读 + 写）" | `hello` **只写不读**。那个后缀是报告模板里写死的，对只写的算子就是错的。已去掉 |
