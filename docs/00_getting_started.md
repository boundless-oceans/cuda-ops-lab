# 00 起步：工具链与第一个 kernel

> 这一章不写算子，只解决一件事：**让"改一行 → 跑一次 → 看到数字变化"这个回路转起来。**
> 回路转不起来，后面十章都是纸上谈兵。

---

## 1. 先确认环境

```bash
conda env create -f environment.yml    # 环境名 opslab
conda activate opslab
python scripts/check_env.py
```

体检报告里最该看的是 `[5/7] GPU 与性能上限` 一节：

```
设备      NVIDIA GeForce RTX 4060 Laptop GPU
算力      8.9  →  编译目标 sm_89
SM 数量   24
理论带宽  ~256 GB/s
Machine Balance  ~54 FLOP/Byte
```

**`Machine Balance` 是这本书的第一个核心概念**，见下一节。

> 为什么不用 `torch.utils.cpp_extension.load()`：环境里的 conda `ninja` 包**不含
> Python 模块**，而且 ninja 可执行文件不一定在 `PATH` 上。所以本仓库自研
> `_build.py` 生成 `build.ninja`，参数全部取自 `build_config`（与体检同一份来源）。

---

## 2. 三个必须先建立的概念

### 2.1 算术强度与 machine balance

```
算术强度 I = FLOP 数 / 搬运字节数        [FLOP/Byte]
machine balance = 峰值算力 / 峰值带宽     [FLOP/Byte]
```

- `I < machine balance` → **memory-bound**：时间花在等内存，优化目标是打满带宽
- `I > machine balance` → **compute-bound**：优化目标是减少运算

本机 machine balance ≈ **54**。而 `elementwise` 的 `add` 算下来 `I = 0.0833` ——
差了**几百倍**。所以它的优化方向从一开始就定死了：**别想着少算，想着快搬。**

### 2.2 理论下限

```
理论最短耗时 = 搬运字节数 / 峰值带宽
```

`add` 在 N = 2²⁴ 时：`201.3 MB ÷ 256 GB/s = 786.3 µs`。

**这是你唯一的判据。** 实测 800 µs 说明到顶了，别再优化；实测 3000 µs 说明还有 4 倍空间。
没有这个数，800 GB/s 摆在面前你不知道该高兴还是该继续。

### 2.3 延迟与并发度（Little's law）

```
需要的在飞请求数 = 访存延迟 × 带宽
```

显存延迟几百个周期，带宽又是几百 GB/s —— 两者相乘，意味着**同时在飞的访存请求
必须有成千上万个**。这个数从哪来？两条路：

| 来源 | 做法 | 本章例子 |
|---|---|---|
| 线程数 × 每线程请求数 | 提高 occupancy，或每线程处理多个元素 | 第 2 章 `v2_unroll4` |
| 指令级并行（ILP） | 每个线程一次发出多个互不依赖的 load | 第 2 章 `v2` 的"先全 load 再统一 store" |

**关键推论：occupancy 低不等于性能差。** 只要每线程的在飞请求够多，线程少一点没关系。
第 1 章的实验就是验证这件事——而且实测结果比预期更有意思（见那一章的 README）。

---

## 3. 工具链：五个你会反复用到的命令

| 工具 | 用途 | 需要 GPU |
|---|---|---|
| `nvcc -Xptxas -v` | 编译并输出**寄存器 / spill / 常量内存**用量 | 否 |
| `cuobjdump -sass` | 看真实生成的**机器指令**（验证向量化、共享内存） | 否 |
| `cuobjdump -res-usage` | 静态资源概览（注意：**看不到动态共享内存**） | 否 |
| `ncu` | 硬件计数器：带宽、stall 原因、occupancy 实况 | **是** |
| `compute-sanitizer` | 越界 / 竞态 / 未初始化内存 | **是** |

本仓库把它们包成了两条命令：

```bash
python python/ops_lab/_build.py --clean -v   # 编译 + ptxas 报告（-v 必需，否则增量构建不重编、无输出）
python scripts/resource_report.py            # 全部 kernel 的资源表（不需要 GPU）
```

---

## 4. 跑通第一个 kernel

```bash
python python/ops_lab/_build.py --clean -v
```

第一次跑会看到类似这样的输出（这已经是一次完整的编译 + 静态分析了）：

```
[1/2] NVCC kernels/01_execution/hello.cu
ptxas info    : Compiling entry function 'hello_v1_naive_kernel' for 'sm_89'
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, 368 bytes cmem[0]
[2/2] LINK build/ext/ops_lab_ext.so
```

然后从 Python 里调用它：

```python
import sys; sys.path.insert(0, "python")
import ops_lab

ext = ops_lab.ext()          # 缺失时自动构建
print(ext.list_kernels())    # 列出全部已登记算子
print(ext._probe())          # 扩展的编译期信息：ABI / 目标架构 / 编译器
```

**`_probe()` 里的 `abi_cxx11` 必须是 0** —— torch 2.0.1 是用旧 ABI 编译的，
这里对不上，`import` 时会报 `undefined symbol`。

---

## 5. 一条最重要的纪律：先写预测

第 1 章的 `bandwidth_probe` 实验，我在写代码之前就写下了预期：

> occupancy 从 ~100% 压到 ~33%，**带宽基本不变**（Little's law）。

实测结果是：**0/24K/32K 三档持平，但 48K（占用率最低）反而快了 5%**。

预测错了一半——**而这一半的错比全对更有价值**，因为它指向了一个没被理解的现象。
如果没写预测，我只会看到"四档都差不多"然后翻页过去。

**所以每一章都先写预期、再回填实测。预测错了不改预测，改文档。**

---

## 6. 下一步

| 想做的事 | 去哪 |
|---|---|
| 看懂一个算子的代码长什么样 | [operator_code_navigation.md](operator_code_navigation.md) |
| 判断一个算子快不快、瓶颈在哪 | [operator_performance_analysis.md](operator_performance_analysis.md) |
| 深入性能模型（Roofline / occupancy） | [01_performance_model.md](01_performance_model.md) |
| 看一条完整的优化阶梯怎么走 | [第 2 章](../kernels/02_elementwise/README.md) |
| 看一个"实验失控"的诚实记录 | [第 1 章](../kernels/01_execution/README.md) |
