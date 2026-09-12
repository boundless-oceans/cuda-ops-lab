// bindings/registry.h
//
// 导出算子的登记表。
//
// 存在的意义：S4 要离线校验"Python 元数据里的算子名 ↔ 扩展实际导出的符号"
// 是否一一对应。如果绑定和登记是两处手写，它们迟早会对不上，那么校验本身
// 就成了新的 bug 来源 —— 所以两者被合成一个动作，见 bind_helpers.h。
//
// 这一层属于 bindings/，可以依赖 torch / pybind11；
// 纯 C++ 的部分（本文件与 registry.cpp）刻意不引入 pybind11。
#pragma once

#include <string>
#include <vector>

namespace ops_lab {

struct KernelInfo {
  std::string name;         // 导出名，如 "elementwise_add_v1_naive"
  std::string chapter;      // 章节，如 "02_elementwise"
  std::string variant;      // 变体，如 "v1_naive"
  std::string description;  // 一行中文说明
};

class Registry {
 public:
  static Registry& instance();

  // 登记一个导出算子。重名会抛异常 —— 重名几乎总是复制粘贴留下的错误，
  // 而且后果（后注册的覆盖先注册的）非常难查，所以宁可在加载时就炸。
  void add(KernelInfo info);

  const std::vector<KernelInfo>& all() const { return entries_; }

 private:
  std::vector<KernelInfo> entries_;
};

}  // namespace ops_lab
