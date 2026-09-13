# 怎么读一个算子仓库（代码导航）

> **姊妹篇**：`operator_performance_analysis.md` 讲"**这个算子快不快、瓶颈在哪**"。
> 这份讲"**这个算子由哪些文件组成、每个文件干什么**"。
>
> 打开一个陌生的算子项目时，通常是这份先派上用场 —— 你总得先知道该打开哪个文件。

本文以 **本仓库第 2 章的 `add`** 作为完整示例。选它是因为它是最"标准"的算子
（有多个变体、有绑定、有元数据），第 1 章不合适 —— 那一章是两个受控实验，
不是标准算子，拿它当模板会误导。

---

## 1. 先破除一个误解：一个算子可以只有一个文件

这是最小形态，可以直接用 `nvcc` 编译运行：

```cuda
// add.cu —— 算子的完整骨架（省略了显存分配与错误检查，只为看清结构）
__global__ void add_kernel(const float* x, const float* y, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = x[i] + y[i];        // ← 整个算子真正"算"的只有这一行
}

// host 侧 launcher：唯一职责是决定 grid / block 怎么分，然后启动
void add(const float* x, const float* y, float* out, int n, cudaStream_t stream) {
  const int block = 256;
  add_kernel<<<(n + block - 1) / block, block, 0, stream>>>(x, y, out, n);
}
```

**算子的本体就是那个 `__global__` 函数加一个 host 侧 launcher。** 就这么多。

一个 `.cu` 文件加一个 `main` 就能编出来跑（`nvcc -arch=sm_89 add.cu -o add`）。
上面省略的是 `main` 里分配显存、填数据、取回结果的部分 —— 那些是**驱动程序**的活，
不是算子的本体。这也是为什么本仓库把它们分到了不同的层。

那为什么真实项目里会变成几十个文件？因为下面四件事，单文件解决不了：

| 要解决的问题 | 结果 |
|---|---|
| 谁来分配显存、把数据搬上 GPU、把结果拿回来 | → 需要**框架适配层**（绑定） |
| 谁来告诉我算得对不对 | → 需要**参考实现 + 测试** |
| 谁来测我有多快、离硬件上限多远 | → 需要**基准 + 理论值** |
| 谁来在 Python / 别的语言里调用我 | → 需要**导出与构建系统** |

**所以目录结构不是审美，是被这四件事逼出来的。** 看到一堆文件时，先问
"它在解决上面哪一件事"，就不会迷路。

---

## 2. 本仓库的整体分层与数据流

**依赖方向是单向的**，这是整个结构的骨架：

```
        ┌─────────────────────────────────────┐
        │  kernels/   纯 CUDA，只认指针和长度   │  ← 零 torch 依赖
        └──────────────┬──────────────────────┘
                       │ 被调用
        ┌──────────────▼──────────────────────┐
        │  bindings/  校验 / 分配 / 取 stream  │  ← 唯一 include torch 的地方
        └──────────────┬──────────────────────┘
                       │ 导出符号
        ┌──────────────▼──────────────────────┐
        │  python/ops_lab/  用户 API + 元数据  │
        └─────────────────────────────────────┘

        native/   纯 CUDA 可执行，复用 kernels/，完全不经过上面两层
        tests/ bench/ scripts/   围着上面这些转的工具
```

**为什么 kernel 层不许出现 torch 符号**，三条实际好处：

1. 编译快（只编 CUDA，不拉几千行 torch 头文件）、报错干净
2. `native/` 可以不加修改地复用同一份 kernel
3. 和 Ascend C 同构 —— Ascend C 的 kernel 同样只认 GM 指针，
   框架适配是另一层的事

### 2.1 一次调用里，数据经过了哪些地方

以 `out = ops_lab.ext().elementwise_add_v1_naive(x, y)` 为例。

**这张图对任何算子都成立**（层与层的职责是固定的），只有最下面的显存布局
是 `add` 特有的。

```
【① Python 侧】
    out = ext.elementwise_add_v1_naive(x, y)
    传下去的是 Tensor 对象：形状 / dtype / 设备 / 数据指针
        │
        ▼
════════════ 以下在【宿主机 CPU】上执行 ════════════
【② bindings/bind_02_elementwise.cpp】
    ⓐ TORCH_CHECK        在 CUDA 上吗？float32 吗？形状一致吗？
    ⓑ x.contiguous()     非连续时，在这里复制出一份新显存   ★ 会搬数据
    ⓒ empty_like(xc)     分配输出的显存                     ★ 会分配
    ⓓ getCurrentCUDAStream()
    ⓔ data_ptr<float>()  ──► 裸指针
        │
        │  ★ 分界线：从这里往下只有指针和长度，不再有 torch
        ▼
【③ kernels/02_elementwise/elementwise_v1_naive.cu】
    ⓕ grid = ceil_div(n, 256)，并检查是否超过 gridDim.x 上限
    ⓖ kernel<<<grid, 256, 0, stream>>>(x_ptr, y_ptr, out_ptr, n)
        │
        │  ★ 异步：CPU 在这里立刻返回，不等 GPU
        ▼
════════════ 以下在【设备 GPU】上执行 ════════════
【④ 每个线程】
        读 x[i] ─┐
        读 y[i] ─┼──► 相加 ──► 写 out[i]
                ┘
        │
        ▼
【⑤ 回到 Python】
    return out          ⚠ 此刻结果很可能还没算完
```

显存里发生的事（以 `N = 8` 为例）：

```
   x    :  [x0] [x1] [x2] [x3] [x4] [x5] [x6] [x7]
   y    :  [y0] [y1] [y2] [y3] [y4] [y5] [y6] [y7]
   out  :  [o0] [o1] [o2] [o3] [o4] [o5] [o6] [o7]
             ↑
             └─ thread 0 读写第 0 个，thread 1 读写第 1 个，依此类推
                相邻线程访问相邻地址 → 一个 warp 的 32 次访问连成 128 字节
                → 这就是"合并访存"
```

**从这张图能读出三件事**：

1. **每次调用固定是 2 读 1 写**（读 `x`、读 `y`、写 `out`）。
   这就是 `bytes = 12N` 的来源 —— 怎么用它算出理论最短耗时，
   见 `operator_performance_analysis.md` 第 0 步。

2. **最多会多出两次显存搬运**：ⓑ `contiguous()`（输入非连续时）和
   ⓒ `empty_like()`（分配输出）。两件事都发生在**② 绑定层**。
   所以"输入是不是连续的"这种问题，答案永远在绑定层，不在 kernel 里。

3. **异步边界在 ⓖ**。`kernel<<<>>>` 返回时 GPU 还没开始跑，Python 拿到的
   `out` 是"将来会有结果"的句柄。**这一点在计时时最要命**：不加同步的计时
   测的是"启动 kernel 花了多久"，而不是"kernel 跑了多久"。
   真正等待发生在 `out.cpu()` / `.item()` / `torch.cuda.synchronize()`。

---

## 3. 基础设施文件：每个算子都要用，但不属于任何算子

### 3.1 `kernels/common/` —— kernel 层的公共工具

| 文件 | 职责 |
|---|---|
| `cuda_check.h` | CUDA 错误检查宏；grid 尺寸计算与**上限判断**（超 2³¹−1 时抛清晰错误）；16 字节对齐判断 |
| `device.h` | 设备查询（SM 数、共享内存、寄存器）；由显存频率与位宽**推算理论峰值带宽**；查询某 kernel 的理论 occupancy |
| `benchmark.h` | `cudaEvent` 计时，返回 median / min / mean；把耗时换算成 GB/s 和"占峰值百分比" |

这三个文件只依赖 CUDA Runtime，不依赖 torch —— 所以 `native/` 也能用。

### 3.2 `bindings/` —— 框架适配层的基础设施

| 文件 | 职责 |
|---|---|
| `registry.h` / `registry.cpp` | 导出算子的**登记表**（名字 / 章节 / 变体 / 描述）。重名会抛异常 |
| `bind_helpers.h` | 把"绑定到 Python"和"登记进表"**合成一个动作**的模板函数 |
| `module.cpp` | Python 扩展的入口（`PYBIND11_MODULE`），挂上各章的绑定，并提供 `list_kernels()` |

`bind_helpers.h` 的存在理由值得单独说：如果绑定和登记是两处手写，
迟早会出现"登记了但没绑"或"绑了但没登记"，那么测试里的一致性校验本身
就成了 bug 来源。合成之后，**漏绑在结构上不可能发生**。

### 3.3 `python/ops_lab/` —— Python 侧基础设施

| 文件 | 职责 |
|---|---|
| `__init__.py` | 包入口，对外暴露 `ext()` / `list_kernels()` / `is_built()` |
| `build_config.py` | **构建参数的唯一来源**：探测 nvcc / gcc / ninja / pybind11、目标架构、ABI、include 与链接参数 |
| `_build.py` | 自研 ninja 构建：生成 `build.ninja` → 编译 → 链接出 `ops_lab_ext.so` |
| `_extension.py` | 按路径加载扩展；发现 `.so` 不存在时自动构建一次 |
| `envguard.py` | 剔除 `sys.path` 里"外来"的路径（见下方说明） |

两个文件的存在理由不显然，值得记一句：

- **`build_config.py`** 被 `scripts/check_env.py` 和 `_build.py` **共用**。
  否则"报告里写的编译参数"和"真正编译用的参数"会各说各话，报告就成了装饰品。
- **`envguard.py`**：某些机器的 shell 会通过 `PYTHONPATH` 注入别的工具链的包目录，
  那些目录属于另一个 Python 版本。混进 `sys.path` 后，同一个脚本在不同终端里
  行为不同，且报错指向的位置看起来完全无辜。所以在入口脚本最早处剔除。

### 3.4 `scripts/`

| 文件 | 职责 |
|---|---|
| `check_env.py` | 环境体检：Python / torch / CUDA / 宿主工具链 / GPU 与性能上限，最后给出推荐编译参数 |
| `resource_report.py` ⬜ | ptxas 离线资源报告（寄存器 / spill / smem），**不需要 GPU 也不需要重新构建** |

（这些脚本具体怎么用、该看哪些数字，见 `operator_performance_analysis.md` 的工具速查。）

---

## 4. 一个算子的完整文件清单（以第 2 章 `add` 为例）

### 4.1 目录树

```
kernels/02_elementwise/
├── elementwise.cuh                ✅  所有变体的 launcher 声明 + 常量
├── elementwise_ops.cuh            ⬜  算子的数学定义（add / relu / sigmoid 的 device 函数）
├── elementwise_v0_uncoalesced.cu  ⬜  变体 0：反面教材（故意写错）
├── elementwise_v1_naive.cu        ✅  变体 1：一线程一元素
├── elementwise_v2_unroll4.cu      ⬜  变体 2：每线程 4 元素
├── elementwise_v3_grid_stride.cu  ⬜  变体 3：grid-stride 循环
├── elementwise_v4_vectorized.cu   ⬜  变体 4：float4 向量化
└── README.md                      ⬜  六段式文档（问题定义 / 理论下限 / 阶梯表 / ...）

bindings/
└── bind_02_elementwise.cpp        ✅  这一章的绑定：校验 → 分配 → 取 stream → 启动

python/ops_lab/
└── elementwise.py                 ⬜  这一章的元数据：OPS 描述符 + VARIANT_NOTES
```

✅ 已实现　⬜ 计划中（本文写于第 2 章阶梯尚未铺完的时候，所以大片是 ⬜）

**变体为什么要一个文件一个**：相邻两个文件的 diff 就是那一级"改了什么"。
这比在一个文件里堆五个函数更容易看出差别，也更容易被 `git log -p` 讲清楚。

### 4.2 每个文件干什么

| 文件 | 职责 |
|---|---|
| `elementwise.cuh` | 声明所有变体的 launcher，签名统一为 `[输入指针] [输出指针] [尺寸] cudaStream_t`。**只有声明，没有实现** |
| `elementwise_ops.cuh` | 算子的数学本体：`add` / `relu` / `sigmoid` 的 `__device__` 函数。三个算子共用同一套 kernel 模板，所以"3 个算子"不等于 3 倍代码量 |
| `elementwise_vN_*.cu` | **一个变体的完整实现**：`__global__` kernel + host launcher。变体之间只差那一级的优化手段 |
| `bind_02_elementwise.cpp` | 把 C++ 函数暴露给 Python。**只做四件事**：校验参数 → 分配输出 → 取 stream → 启动 kernel。一行计算逻辑都没有 |
| `elementwise.py` | 声明这一章的元数据：有哪些算子、每个算子有哪些变体、参考实现是什么、用哪些 shape 测、搬运多少字节。测试和基准脚本都从这里读，**新增算子不需要改任何中心文件** |
| `README.md` | 这一章的学习文档，六段式模板 |

### 4.3 通用骨架：任何一章都长这样

```
kernels/<NN>_<章节名>/          这一章的 kernel（纯 CUDA）
bindings/bind_<NN>_<章节名>.cpp 这一章的绑定
python/ops_lab/<章节名>.py      这一章的元数据
```

新增一个算子，就是在这三处各加东西。

---

## 5. 打开一个**陌生**算子仓库的阅读顺序

不要从构建脚本开始看（那是最不重要的部分）。按这个顺序：

| 顺序 | 看什么 | 回答什么问题 |
|---|---|---|
| 1 | `README` | 这仓库是干什么的、算什么算子 |
| 2 | **找 `__global__`** | 算子的本体在哪。这是唯一真正"算"的地方 |
| 3 | 同一个文件里的 host launcher | grid/block 怎么定的？边界怎么处理的？ |
| 4 | 找框架适配层（`bind*` / `csrc/` / `torch.extension`） | 参数怎么从 Python 进来的？做了什么校验？ |
| 5 | 找测试和基准 | 作者怎么判断对错、怎么测快慢 |
| 6 | 最后才看 `setup.py` / `CMakeLists.txt` / 构建脚本 | 怎么编出来的 |

第 2 步是关键：**一个算子仓库里可能有一百个函数，但只有标着 `__global__` 的那些
是在 GPU 上真正跑的东西。** 其余的（launcher、绑定、分配）都是围着它转的服务代码。

---

## 6. 新增一个算子要动哪些文件

当成施工检查表用，防止漏：

| # | 动作 | 位置 |
|---|---|---|
| 1 | 声明 launcher | `kernels/<章节>/<算子>.cuh` |
| 2 | 写 kernel 与 launcher | `kernels/<章节>/<算子>_vN_<技术>.cu` |
| 3 | 写绑定函数（校验/分配/取 stream/启动） | `bindings/bind_<章节>.cpp` |
| 4 | 用 `bind_kernel(...)` 登记并导出 | 同上 |
| 5 | 声明元数据（变体 / 参考实现 / shape / bytes / flops） | `python/ops_lab/<章节>.py` |
| 6 | 写这一级的"思路"一句话 | 同上（`VARIANT_NOTES`） |
| 7 | 跑一次构建与测试 | `python python/ops_lab/_build.py` |

第 4 步和第 5 步是**最容易漏**的地方：kernel 写了、也绑了，但元数据里忘了登记，
结果测试和基准都看不到它。这就是"绑定与登记合一"要解决的问题。

---

## 7. 不要以为所有项目都长这样

本仓库的结构是**为了"教学 + 可离线验证"**而设计的，属于比较干净的一类。
真实项目差异很大：

| 形态 | 典型结构 | 特点 |
|---|---|---|
| **最小 demo** | 一个 `.cu` 文件 | 没有绑定、没有测试。适合验证想法，不适合长期维护 |
| **PyTorch 扩展**（多数开源算子库） | `csrc/xxx.cpp`（绑定）+ `csrc/xxx_kernel.cu`（kernel）+ `setup.py` | 和本仓库的分层思路相同，但通常不会把 kernel 抽成"零框架依赖" |
| **框架内置算子**（如 PyTorch 的 ATen） | 算子声明（`native_functions.yaml`）、公共头、CPU 实现、CUDA 实现、注册、测试 —— 分散在四五个目录 | 新增一个算子要动 5~6 个文件，因为要同时支持多种后端与自动微分 |
| **CANN / Ascend C 算子工程** | host 侧（tiling、形状推导）与 kernel 侧分开 | 分法与本仓库的 `bindings/` + `kernels/` 是同一个道理。<br>**具体结构本文不写** —— 我尚未核实，留到 M6 的 `03_cuda_to_ascendc.md` 里确认后再写 |

**判断一个项目的分层好不好，问四个问题**：

1. 能不能**只编译 kernel**，不拉框架依赖？
2. 能不能**不经过 Python** 就跑？（本仓库的 `native/`）
3. 新增一个算子要改几个文件？有没有"必须记得同时改两处"的地方？
4. 有没有**同一个信息写在两个地方**？（迟早会不一致）

---

## 一句话总结

**算子的本体只有一个 `__global__` 函数；所有其它文件都在解决四件事：
分配显存、验证正确、测量性能、被框架调用。**
看陌生仓库时先找 `__global__`，再看 launcher 怎么定 grid，最后才关心构建脚本。
