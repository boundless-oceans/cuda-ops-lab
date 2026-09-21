// kernels/03_reduction/reduction_init.cu
//
// 把归约输出写成单位元。所有变体在启动主 kernel **之前**都要先调它。
//
// 为什么要单独一个小 kernel，而不是让主 kernel 的某个线程去写：
//   * 主 kernel 有几百个 block，谁先跑到谁写就成了竞态；写成"block 0 先写"
//     还需要一次全局同步，而 CUDA 没有便宜的全局屏障。
//   * 这是**一次启动**的固定成本（grid=1, block=1，实测在计时噪声以下），
//     换来的是"输出永远有正确的初值"这条简单到不会出错的性质。
//
// 另一个容易被忽略的作用在 launcher 的空输入分支里：n == 0 时主 kernel 不启动，
// 但输出仍然必须被写出来。把这一步放在这里，空输入就自动正确了。
#include "kernels/03_reduction/reduction.cuh"

namespace ops_lab {
namespace reduction {
namespace {

__global__ void init_scalar_kernel(float* out, float identity) { *out = identity; }

}  // namespace

void init_reduction_output(float* out, float identity, cudaStream_t stream) {
  init_scalar_kernel<<<1, 1, 0, stream>>>(out, identity);
  OPSLAB_CUDA_CHECK_LAUNCH();
}

}  // namespace reduction
}  // namespace ops_lab
