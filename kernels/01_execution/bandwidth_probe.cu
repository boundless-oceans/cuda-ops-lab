// kernels/01_execution/bandwidth_probe.cu
//
// 实验二：occupancy 受控实验。
//
// 核心设计：四个变体的**拷贝逻辑完全相同**，唯一变量是动态共享内存占用。
// 因此带宽的任何变化都只能归因于 occupancy，而不是"代码写得好不好"。
#include "kernels/01_execution/execution.cuh"

#include <stdexcept>

#include "kernels/common/cuda_check.h"

namespace ops_lab {
namespace execution {
namespace {

// 变体 1..4 分别使用的动态共享内存字节数。
// 48 KB 正好是"无需 cudaFuncSetAttribute 申请 opt-in"的上限。
constexpr int kSmemBytes[kProbeVariantCount] = {0, 24 * 1024, 32 * 1024, 48 * 1024};

// 拷贝主体：float4 主循环 + 不足 4 个元素的标量尾巴。
//
// 抽成 device 函数，是为了让"带 smem / 不带 smem"两份 kernel 共享同一份逻辑,
// 从而保证两个变体之间**唯一的差别**就是共享内存。
__device__ __forceinline__ void copy_body(const float4* __restrict__ x4,
                                          float4* __restrict__ out4,
                                          const float* __restrict__ x,
                                          float* __restrict__ out,
                                          int64_t n4,
                                          int64_t n) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4;
       i += stride) {
    out4[i] = x4[i];
  }
  // 尾巴：n - 4*n4 < 4，所以只有第一个 block 的几个线程会命中。
  const int64_t j = n4 * 4 + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (j < n) {
    out[j] = x[j];
  }
}

// 变体 1：不声明 extern __shared__。
__global__ void copy_plain_kernel(const float4* __restrict__ x4, float4* __restrict__ out4,
                                  const float* __restrict__ x, float* __restrict__ out,
                                  int64_t n4, int64_t n) {
  copy_body(x4, out4, x, out, n4, n);
}

// 变体 2/3/4：声明并使用动态共享内存。
__global__ void copy_smem_kernel(const float4* __restrict__ x4, float4* __restrict__ out4,
                                 const float* __restrict__ x, float* __restrict__ out,
                                 int64_t n4, int64_t n) {
  extern __shared__ float4 smem[];
  copy_body(x4, out4, x, out, n4, n);

  // 必须**真的碰一下**这段共享内存。动态 smem 虽然是启动参数，但若编译器把整个
  // extern __shared__ 声明优化掉，就无法保证它真的被分配 —— 实验会静默失效
  // （看起来申请了 48KB，实际是 0）。下面这两步编译不掉。
  __syncthreads();
  if (threadIdx.x == 0 && n4 > 0) {
    smem[0] = out4[0];
  }
  __syncthreads();
  if (threadIdx.x == 0 && n4 > 0 && smem[0].x == 1e30f) {
    out4[0] = smem[0];  // 永不成立，但编译器不能假设它不成立
  }
}

int smem_for(int variant) {
  if (variant < 1 || variant > kProbeVariantCount) {
    throw std::invalid_argument("bandwidth_probe: variant 必须在 1.." +
                                std::to_string(kProbeVariantCount) + " 之间");
  }
  return kSmemBytes[variant - 1];
}

const void* kernel_for(int variant) {
  return variant == 1 ? reinterpret_cast<const void*>(copy_plain_kernel)
                      : reinterpret_cast<const void*>(copy_smem_kernel);
}

}  // namespace

void bandwidth_probe(const float* x, float* out, int64_t n, int variant, cudaStream_t stream) {
  const int smem = smem_for(variant);
  if (n <= 0) {
    return;
  }

  const int64_t n4 = n / 4;
  // 至少 1 个 block：n < 4 时主循环无事可做，但尾巴仍需要有人处理。
  const int64_t grid = grid_for(n4, kBlockSize, /*allow_grid_stride=*/true);

  const float4* x4 = reinterpret_cast<const float4*>(x);
  float4* out4 = reinterpret_cast<float4*>(out);

  if (variant == 1) {
    copy_plain_kernel<<<static_cast<unsigned int>(grid), kBlockSize, 0, stream>>>(x4, out4, x, out,
                                                                                  n4, n);
  } else {
    copy_smem_kernel<<<static_cast<unsigned int>(grid), kBlockSize, smem, stream>>>(x4, out4, x,
                                                                                    out, n4, n);
  }
  OPSLAB_CUDA_CHECK_LAUNCH();
}

int bandwidth_probe_smem_bytes(int variant) { return smem_for(variant); }

Occupancy bandwidth_probe_occupancy(int variant, int block_size) {
  return query_occupancy(kernel_for(variant), block_size, smem_for(variant));
}

}  // namespace execution
}  // namespace ops_lab
