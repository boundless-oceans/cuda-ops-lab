// bindings/module.cpp
//
// Python 扩展的入口。
//
// 层层递进的故障定位（S1 分三步做的原因）：
//
//   S1-1  只编译 .cu              → 失败说明问题在 nvcc 侧
//   S1-2  编译 .cpp + 链接 + 加载  → 失败说明问题在构建管道 / 动态库加载
//   S1-3  引入 torch::Tensor       → 失败就一定是 ABI 或 torch 符号解析
//
// 本文件自己不引用任何 torch 符号，只负责把各章的绑定挂上来。
#include <pybind11/pybind11.h>

#include <string>

#include "bindings/registry.h"

namespace py = pybind11;

namespace ops_lab {

// 各章的绑定的入口，实现在 bind_<chapter>.cpp 里。
void bind_elementwise(py::module_& m);

}  // namespace ops_lab

namespace {

// 编译期 ABI 设置（由 build_config 注入 -D_GLIBCXX_USE_CXX11_ABI=0/1）
#if defined(_GLIBCXX_USE_CXX11_ABI)
constexpr int kAbiCxx11 = _GLIBCXX_USE_CXX11_ABI;
#else
constexpr int kAbiCxx11 = -1;  // 未定义，说明构建脚本没注入，属异常情况
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

// 列出所有已登记的导出算子。
// S4 会用它做"元数据 ↔ 导出符号"的双向一致性校验。
py::list list_kernels() {
  py::list out;
  for (const auto& k : ops_lab::Registry::instance().all()) {
    py::dict entry;
    entry["name"] = k.name;
    entry["chapter"] = k.chapter;
    entry["variant"] = k.variant;
    entry["description"] = k.description;
    out.append(entry);
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(ops_lab_ext, m) {
  m.doc() = "cuda-ops-lab 的 CUDA 扩展（手写 kernel 的 Python 绑定）";

  m.def("_probe", &probe,
        "构建探针：返回扩展的编译期信息，用于确认加载的是正确的构建产物");
  m.def("list_kernels", &list_kernels,
        "列出所有已登记的导出算子（name/chapter/variant/description）");

  // 各章注册自己的绑定
  ops_lab::bind_elementwise(m);
}
