// kernels/common/cuda_check.h
//
// kernel 层的错误检查。
//
// 为什么不用 C10_CUDA_CHECK / AT_CUDA_CHECK？
//   因为 kernels/ 目录的铁律是「不出现任何 torch 符号」。
//   这里的头文件只依赖 CUDA Runtime，所以 native/ 的纯 CUDA 可执行文件
//   也能直接复用同一份 kernel（design §5.1）。
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ops_lab {

// cudaError_t -> 人类可读字符串（含错误名 + 描述）
inline std::string cuda_error_string(cudaError_t err) {
  return std::string(cudaGetErrorName(err)) + ": " + cudaGetErrorString(err);
}

// 检查一次 CUDA API 调用。失败时抛 std::runtime_error，pybind11 会把它
// 翻译成 Python 的 RuntimeError。
inline void cuda_check_impl(cudaError_t err, const char* expr, const char* file, int line) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA 调用失败 ") + file + ":" + std::to_string(line) +
                             "\n  表达式: " + expr + "\n  错误  : " + cuda_error_string(err));
  }
}

// 用法：OPSLAB_CUDA_CHECK(cudaMalloc(&p, n));
#define OPSLAB_CUDA_CHECK(expr) \
  ::ops_lab::cuda_check_impl((expr), #expr, __FILE__, __LINE__)

// kernel 启动后立刻检查一次。注意：kernel 内的越界/非法访问是异步的，
// 这个宏只能抓到"启动参数非法"（例如 grid 超过 2^31-1）这类同步错误。
#define OPSLAB_CUDA_CHECK_LAUNCH() OPSLAB_CUDA_CHECK(cudaGetLastError())

// ---------------------------------------------------------------- 尺寸工具

// 向上取整除法。注意 a 可能很大（int64），所以先在 int64 里算。
inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

// grid 维度上限：gridDim.x 是 unsigned int，最大 2^31-1。
// 超过这个值的 n 必须走 grid-stride 循环（第 2 章 v3 会讲这件事）。
constexpr int64_t kMaxGridDimX = 2147483647LL;

// 计算恰好覆盖 n 个元素所需的 block 数，并校验不会溢出 grid 上限。
// allow_grid_stride=false 时，超出上限直接抛异常（这是 v1/v2/v4 的行为，
// 目的是让"grid 有上限"这件事在测试里可见，而不是静默出错）。
inline int64_t grid_for(int64_t n, int64_t block, bool allow_grid_stride = false) {
  const int64_t g = ceil_div(n, block);
  if (g > kMaxGridDimX) {
    if (!allow_grid_stride) {
      throw std::runtime_error(
          "grid 维度溢出：n = " + std::to_string(n) + "，block = " + std::to_string(block) +
          " 需要 " + std::to_string(g) + " 个 block，超过 gridDim.x 上限 " +
          std::to_string(kMaxGridDimX) +
          "。\n  这个变体不支持超大 n；请用 grid-stride 变体（v3_grid_stride）。");
    }
    return kMaxGridDimX;
  }
  return g < 1 ? 1 : g;  // 至少 1 个 block：n == 0 时也允许启动空 kernel
}

// 指针是否满足 16 字节对齐（float4 的前提条件）
inline bool is_aligned_16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

}  // namespace ops_lab
