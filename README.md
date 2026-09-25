# CUDA 算子优化阶梯
### 从零手写算子 → 摸到硬件上限 → 迁移到 Ascend C

手写 CUDA 算子的学习仓库。每个算子写成一条**优化阶梯**：从**故意写错**的反面教材
一路爬到接近硬件上限，每一级都对着**实测数字**说明它改了什么、值不值。

最终目的不是攒一堆 kernel，而是把同一套思维迁移到 Ascend C（CANN 算子赛）。

---

## 先看结果：这条阶梯被自己的数据推翻了一半

`elementwise` 一章写了 `out = x + y` 的五个版本。原计划是一条从 15% 爬到 95% 的
阶梯，实测（RTX 4060 Laptop，N = 2²⁴，理论峰值 256 GB/s）：

| 变体 | 思路 | median (µs) | GB/s | %峰值 |
|---|---|---|---|---|
| `v0_uncoalesced` | 反面教材：相邻线程地址相隔 `gridDim.x` | 3665.4 | 54.9 | **21.5%** |
| **`v1_naive`** | 一线程一元素、已合并访存 | **841.7** | **239.2** | **93.4%** |
| `v2_unroll4` | 每线程 4 元素，用 ILP 换并发度 | 845.8 | 238.0 | 93.0% |
| `v3_grid_stride` | grid 固定 + 循环步进 | 853.0 | 236.0 | 92.2% |
| `v4_vectorized` | `float4`，访存指令数降 4 倍 | 853.0 | 236.0 | 92.2% |
| torch | `x + y`（基线） | 854.5 | 235.6 | 92.0% |

**最值得看的不是那个 93.4%，而是"后面四级没有任何提升"。**

- `v1_naive` 已经把带宽打满，`v2`/`v3`/`v4` 全在 ±2% 内
- **"打满之后继续微调是浪费生命"** —— 这句话被自己的数据证实了
- 顺带：我们的 `v1` 比 `torch` 的 `x + y` 还快 1.5%

还有一些**反直觉**的发现，都写在各章 README 里：

| 发现 | 在哪 |
|---|---|
| 未合并访存的真实代价是 **4.35×**，不是教科书的 32×（L2 接住了复用，瓶颈从 DRAM 带宽移到 L2 事务率） | [第 2 章](kernels/02_elementwise/README.md) |
| **occupancy 最低的那档反而快 5%**：可复现（两条测量路径 + 交错测量都确认），但机制仍未查明 | [第 1 章](kernels/01_execution/README.md) |
| **块内归约的优化测不出来**：同步 8 次 → 1 次、共享内存访问 26 次 → 2 次（SASS 可证），耗时**一动不动** —— 因为按理论下限算，块内那棵树只占 **0.1%** | [第 3 章](kernels/03_reduction/README.md) |
| 而**只用 1 个 block 慢 16.3 倍** —— 这一个变量的权重比上面那条"优化"大三个数量级 | [第 3 章](kernels/03_reduction/README.md) |
| **但"块内优化没用"不能外推**：scan 的块内 work 是每元素 O(log B) 而不是 O(1)，同一类手法第一次能值 **400 µs** | [第 4 章](kernels/04_scan/README.md) |
| **work-efficient 不一定更快**：Blelloch 的总 work 比 Kogge-Stone 少 3 倍，实测却慢 **27%** —— 在 GPU 上"级数 + 发散"比总 work 值钱 | [第 4 章](kernels/04_scan/README.md) |
| **`%峰值` 高不等于快**：12N 做到 90%（393 µs）打不过 8N 做到 85%（308 µs）。判据要看时间，不看百分比 | [第 4 章](kernels/04_scan/README.md) |
| `sigmoid` 带 `exp` 却和 `relu` 带宽**一模一样**——`exp` 被访存延迟完全盖住 | [第 2 章](kernels/02_elementwise/README.md) |

---

## 核心方法

### 0. 先确认你测的是稳态

这一条排在"先算理论下限"**之前**：它不成立的时候，后面所有推理都是自娱自乐。

实测教训：同一个 reduction kernel 被两条独立路径测出 **313.3 µs** 和 **275.5 µs**（差 14%）。
根因不是 kernel，是**显存时钟** —— 冷机 7001 MHz、稳态 8001 MHz，而 `%峰值` 的分母
用的是额定的 8001。把两边换算成"当时那档时钟的利用率"：**95.6% 和 95.7%**，一模一样。

所以基准脚本现在会先预热到时钟稳态、**把测时的时钟写进结果表头**，并**交错测量**
（轮间错开起始变体）抵消慢漂移。详见
[怎么看一个算子](docs/operator_performance_analysis.md) 的"测之前"一节。

### 1. 每级阶梯必须对应一个真实的硬件瓶颈

反面教材不是凑数：`v0` 存在的意义，是让"未合并访存"这件事有一个**可测量的代价**，
而不是一句口号。

### 2. 先算理论下限，再看实测

写代码之前先算：搬多少字节、算术强度多少、和 machine balance 比是 memory-bound
还是 compute-bound、**理论最短耗时是多少**。没有这个数，实测出 800 GB/s 你不知道
该高兴还是该继续优化。

### 3. 预测写在前面，打脸就记下来

每章都先写下预期，再回填实测。**预测错了不改预测，改文档** —— 上面那张表就是例子。

### 4. 到 85% 峰值就停手

然后写清楚"为什么到不了 100%"，把时间花到别处。这条原则被两章数据验证了：
第 2 章的 `v1` 到 93.4% 之后剩下四级都是白费；第 3 章的块内优化（同步 8 次 → 1 次）
干脆**一点时间都没省**——因为它只占理论下限的 0.1%。

---

## 快速开始

需要一台有 NVIDIA GPU 的机器（本项目在 RTX 4060 Laptop / sm_89 上开发）。

```bash
# 1) 环境（conda 环境名 opslab，见 environment.yml）
conda env create -f environment.yml
conda activate opslab

# 2) 环境体检（编译参数、峰值带宽、machine balance 都从这里来）
python scripts/check_env.py

# 3) 构建（自研 ninja，首次会自动触发；参数全部来自 build_config，与体检一致）
python python/ops_lab/_build.py -v

# 4) 测试：符号一致性 + 参考实现比对 + 边界用例
python tests/run_all.py

# 5) 基准：生成优化阶梯表（写到 bench/results/，该目录不入库）
python bench/run_bench.py --chapters elementwise

# 6) 静态资源：寄存器 / spill / 共享内存（不需要 GPU）
python scripts/resource_report.py

# 7) 纯 CUDA 对照线（不依赖 Python / torch，可直接喂给 ncu）
cmake -B build/native -S native && cmake --build build/native -j
./build/native/main_02_elementwise
ncu -k "regex:add_v0_kernel" --launch-count 1 --set full ./build/native/main_02_elementwise
```

一行搞定前面四步：`bash scripts/run_all.sh`

---

## 仓库结构

```
kernels/     纯 CUDA，只认指针和长度        ← 零 torch 依赖
bindings/    校验 / 分配 / 取 stream / 启动  ← 唯一 include torch 的地方
python/      用户 API + 算子元数据 + 参考实现
native/      纯 CUDA 可执行文件（不依赖 Python）
tests/       零依赖测试 harness（环境里没有 pytest）
bench/       基准与阶梯表
scripts/     环境体检、静态资源报告
docs/        专题文档
```

**每个文件是干什么的、为什么这么分**，见 **[docs/operator_code_navigation.md](docs/operator_code_navigation.md)**
——那里是"仓库结构"这件事的唯一出处，本文不再重复。

---

## 文档索引

| 文档 | 什么时候看 |
|---|---|
| [docs/00_getting_started.md](docs/00_getting_started.md) | **第一次上手**：工具链、编译流程、跑通第一个 kernel |
| [docs/operator_code_navigation.md](docs/operator_code_navigation.md) | 不知道某个文件是干什么的 / 想读别人的算子仓库 |
| [docs/operator_performance_analysis.md](docs/operator_performance_analysis.md) | 想判断一个算子快不快、瓶颈在哪 |
| [docs/01_performance_model.md](docs/01_performance_model.md) | Roofline、occupancy、latency hiding 的直觉 |

各章的详细内容在各章目录的 README 里（六段式：问题定义 / 理论下限 / 阶梯表 /
实现要点 / ncu 观察 / Ascend C 对照）。

---

## 现状

**M1 完成**：环境与基础设施、第 0~2 章、测试与基准、静态资源报告、纯 CUDA 对照线。

| 章 | 内容 | 状态 |
|---|---|---|
| 0 | 工具链 + Roofline 入门 | ✅ |
| 1 | 执行模型 + occupancy 实验 | ✅ |
| 2 | elementwise 五级阶梯 | ✅ |
| 3 | reduction 五级阶梯（sum / max） | ✅ |
| 4 | scan 六级阶梯（inclusive prefix sum） | ✅ |
| 5~7 | transpose / softmax / norm | ⬜ |
| 8 | GEMM（比赛核心） | ⬜ |
| 9~10 | attention / 量化 | ⬜ |
| — | Ascend C 落地 | ⬜ |

当前规模：**38 个 kernel**（8 个算子 × 37 个变体）、**82 个测试用例**、
**两条独立的性能测量路径**（Python/torch 与纯 CUDA，本章在 0.5% 内互相印证）。

---

## 已知的坑

这些不是"待办"，是**已经踩过并写下来的**：

- **`nvcc` 不是 gcc 的透明包装**：宿主编译器选项走 `-Xcompiler`，链接器选项走 `-Xlinker`；
  且 `-Xcompiler=-Wl,-rpath,X` 会**静默失效**（该值逗号分隔）
- **`cuobjdump -res-usage` 看不到动态共享内存**：那是启动参数，不是编译期属性
- **测试形状不能拿来跑基准**：工作集装进 L2 就会测成缓存带宽
- **"需要 GPU"的标记会掩盖离线可查的错误**：元数据声明的参数个数与绑定不一致，
  离线就能查，却因为测试标了 `gpu=True` 而被跳过
- **同一份"峰值带宽"公式不该有两处手写**：`device_query.py` 是唯一出处
- **冷机的显存时钟达不到额定值**：实测 7001 → 8001 MHz（差 **14%**）。
  不预热就测，最先测的几个变体会被凭空打上"只有 83% 峰值"的标签。
  **`%峰值` 必须和测时的时钟一起报**，否则那个百分比无法自证

更细的记在各章 README 的"踩坑记录"里。

---

## 非目标

- ❌ 不做通用深度学习框架，不追求算子齐全（不做 conv / batchnorm 全家族）
- ❌ 不追求超越 cuBLAS —— 目标是**理解**为什么快，不是刷榜
- ❌ 主路径不引入 cub / cutlass / triton（它们在"对照"里提及，不参与实现）
- ❌ 不做多卡 / 分布式

---

## LICENSE

BSD-3-Clause，见 [LICENSE](LICENSE)。
