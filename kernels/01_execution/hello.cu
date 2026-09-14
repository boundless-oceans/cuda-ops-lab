// kernels/01_execution/hello.cu
//
// 实验一：把 grid/block/thread 的映射关系变成可以跑、可以看的东西。
#include "kernels/01_execution/execution.cuh"

#include <stdexcept>

#include "kernels/common/cuda_check.h"

namespace ops_lab {
namespace execution {
namespace {

// v1：一个线程负责一个元素，i 只算一次。
__global__ void hello_v1_naive_kernel(int32_t* __restrict__ out, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = static_cast<int32_t>(i);
  }
}

// v2：grid 固定，用循环步进覆盖全部元素。
//   stride 是"整个 grid 一次能覆盖的元素数"，这是 grid-stride 循环的核心。
__global__ void hello_v2_grid_stride_kernel(int32_t* __restrict__ out, int64_t n) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    out[i] = static_cast<int32_t>(i);
  }
}

}  // namespace

void hello_v1_naive(int32_t* out, int64_t n, cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  // allow_grid_stride = false：grid 装不下就抛错，而不是静默出错。
  const int64_t grid = grid_for(n, kBlockSize, /*allow_grid_stride=*/false);
  hello_v1_naive_kernel<<<static_cast<unsigned int>(grid), kBlockSize, 0, stream>>>(out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

void hello_v2_grid_stride(int32_t* out, int64_t n, int block_size, int grid_size,
                          cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  if (block_size <= 0) {
    throw std::invalid_argument("hello_v2_grid_stride: block_size 必须为正数");
  }
  // grid_size <= 0 表示"按设备规模自动选"。
  //
  // 这个约定**必须放在 launcher 里，不能放在绑定层** —— 曾经它只在 Python 绑定里
  // 实现，于是 native 线传 0 就直接抛异常了（同一约定两个调用方，必有一个踩坑）。
  // 现在两边都把这个值原样传进来，由这里统一决定。
  const int grid = grid_size > 0 ? grid_size : default_grid_size(block_size);

  // 这里没有 grid 上限检查 —— 因为循环步进天然覆盖任意大的 n。
  hello_v2_grid_stride_kernel<<<grid, block_size, 0, stream>>>(out, n);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

}  // namespace execution
}  // namespace ops_lab
