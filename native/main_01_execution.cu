// native/main_01_execution.cu
//
// 第 1 章的可执行文件：两个受控实验（hello + occupancy）。
//
//   ./main_01_execution                 # 默认 N = 2^24
//   ./main_01_execution 4194304 200
//
// 这一章不是"优化阶梯"，所以输出不是阶梯表，而是：
//   * hello 两个变体的带宽（差别在"grid 有没有上限"）
//   * bandwidth_probe 四个变体的带宽 + 对应的理论 occupancy
// 后者的看点正是"**占用率掉到 1/3，带宽却不掉**"。
#include <cstdio>

#include "native/bench_main.h"

#include "kernels/01_execution/execution.cuh"

int main(int argc, char** argv) {
  using namespace ops_lab::execution;
  using namespace ops_lab::native;

  const NativeOptions opt = parse_options(argc, argv);

  // ---------------------------------------------------------------- 实验一
  // hello 只写：out[i] = i，4 字节/元素。
  // 注意输出是 **int32**（它算的是索引），不是 float —— 所以用 I32 缓冲。
  const int64_t hello_bytes = opt.n * 4;
  if (!print_header("01_execution / hello", opt, hello_bytes)) {
    return 2;
  }
  const double peak = peak_gbps();

  DeviceBufferI32 out(opt.n);
  bench_row("v1_naive", hello_bytes, [&] { hello_v1_naive(out.get(), opt.n, 0); }, opt, peak);
  bench_row(
      "v2_grid_stride", hello_bytes,
      [&] { hello_v2_grid_stride(out.get(), opt.n, 256, /*grid_size=*/0, 0); }, opt, peak);

  // ---------------------------------------------------------------- 实验二
  // bandwidth_probe 是纯拷贝：读 4N + 写 4N
  const int64_t probe_bytes = opt.n * 8;
  print_header("01_execution / bandwidth_probe（occupancy 受控实验）", opt, probe_bytes);

  DeviceBuffer src(opt.n);
  DeviceBuffer dst(opt.n);
  fill_pattern(src, 3);

  struct ProbeVariant {
    const char* label;
    int index;  // 传给 bandwidth_probe 的 variant 参数，1..4
  };
  const ProbeVariant kProbes[] = {
      {"v1_smem0", 1},
      {"v2_smem24k", 2},
      {"v3_smem32k", 3},
      {"v4_smem48k", 4},
  };

  for (const auto& p : kProbes) {
    bench_row(
        p.label, probe_bytes,
        [&] { bandwidth_probe(src.get(), dst.get(), opt.n, p.index, 0); }, opt, peak);
  }

  // ------------------------------------------------- 理论 occupancy（host 计算）
  std::printf("\noccupancy（host 侧计算，不启动 kernel）\n");
  std::printf("%-14s %13s %11s %14s %11s\n", "variant", "dyn smem(B)", "blocks/SM",
              "warps/SM", "occupancy");
  std::printf("-----------------------------------------------------------------\n");
  for (const auto& p : kProbes) {
    const ops_lab::Occupancy oc = bandwidth_probe_occupancy(p.index, 256);
    std::printf("%-14s %13d %11d %9d/%-4d %10.1f%%\n", p.label, oc.dynamic_smem_bytes,
                oc.blocks_per_sm, oc.active_warps_per_sm, oc.max_warps_per_sm,
                oc.occupancy * 100.0);
  }

  std::printf("\n预期：occupancy 随共享内存增加而下降，但带宽基本不变 ——\n");
  std::printf("      纯流式拷贝在约 1/3 占用率下仍能打满带宽（Little's law：每线程\n");
  std::printf("      多个在飞访存请求可以补偿线程数不足）。\n");
  std::printf("      若实测带宽明显下降，说明 grid 太小、而非 occupancy 不够。\n");
  return 0;
}
