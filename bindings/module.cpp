// bindings/module.cpp
//
// Python 扩展的入口。
//
// S1-2 阶段这里**只有 `_probe()`，故意不引入任何 torch 符号**。
// 目的是把故障二分掉：
//
//   S1-1  只编译 .cu            → 失败说明问题在 nvcc 侧
//   S1-2  编译 .cpp + 链接 + 加载 → 失败说明问题在构建管道 / 动态库加载
//   S1-3  才引入 torch::Tensor   → 失败就一定是 ABI 或 torch 链接的问题
//
// 关于 rpath：这里不引用任何 torch 符号，原本预期链接器（Ubuntu 默认
// --as-needed）会把 -ltorch 之类丢掉、从而验证不到 rpath。
// 但实测并非如此 —— nvcc 与 g++ 都仍把 libtorch.so 记进了 DT_NEEDED
// （本机链接并未启用 --as-needed），所以 rpath 在本步就已经被真实验证：
// ldd 能把 libtorch.so 解析到 torch/lib 目录下。
//
// 于是 S1-3 要验证的是剩下那一半：真正的 torch 符号能否按 ABI=0 正确解析。
#include <pybind11/pybind11.h>

#include <string>

namespace py = pybind11;

namespace {

// 编译期 ABI 设置（由 build_config 注入 -D_GLIBCXX_USE_CXX11_ABI=0/1）
#if defined(_GLIBCXX_USE_CXX11_ABI)
constexpr int kAbiCxx11 = _GLIBCXX_USE_CXX11_ABI;
#else
constexpr int kAbiCxx11 = -1;  // 未定义，说明构建脚本没注入，是异常情况
#endif

// 构建探针：确认扩展**确实被加载进来了**，而不是某个同名的旧文件。
// 这些字段都是编译期常量，运行期零开销，但对排查构建问题很有用。
py::dict probe() {
  py::dict info;
  info["module"] = "ops_lab_ext";
  info["cxx_standard"] = static_cast<long>(__cplusplus);
  info["abi_cxx11"] = kAbiCxx11;
#if defined(OPS_LAB_TARGET_ARCH_NUM)
  info["target_arch_num"] = OPS_LAB_TARGET_ARCH_NUM;
#else
  info["target_arch_num"] = -1;
#endif
#if defined(__GNUC__)
  info["compiler"] = std::string("GCC ") + __VERSION__;
#else
  info["compiler"] = "unknown";
#endif
  return info;
}

}  // namespace

PYBIND11_MODULE(ops_lab_ext, m) {
  m.doc() = "cuda-ops-lab 的 CUDA 扩展（手写 kernel 的 Python 绑定）";

  m.def("_probe", &probe,
        "构建探针：返回扩展的编译期信息，用于确认加载的是正确的构建产物");
}
