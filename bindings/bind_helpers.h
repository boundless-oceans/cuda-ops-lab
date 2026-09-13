// bindings/bind_helpers.h
//
// "绑定 + 登记"合成一个动作的辅助模板。
//
// 为什么合成：S4 要对"Python 元数据里的算子名"与"扩展实际导出的符号"做
// 双向一致性校验。若绑定（m.def）与登记（Registry::add）是两处手写，
// 迟早会出现"登记了但没绑"或"绑了但没登记"，校验本身就成了 bug 来源。
// 合成之后，漏绑在结构上不可能发生。
//
// 注意这里用完整的 `pybind11::` 而不是简写 `py::`：**头文件必须自给自足**。
// 简写依赖调用方先定义 `namespace py = pybind11` —— 而原先包含本文件的两个
// .cpp 都先 include 了 torch/extension.h，torch 的内部头文件恰好定义了这个
// 别名，于是把问题掩盖了，直到第一个不依赖 torch 的绑定文件
// （bind_device.cpp）出现才炸出来。
#pragma once

#include <pybind11/pybind11.h>

#include <string>
#include <utility>

#include "bindings/registry.h"

namespace ops_lab {

template <typename Fn>
void bind_kernel(pybind11::module_& m,
                 const std::string& name,
                 const std::string& chapter,
                 const std::string& variant,
                 const std::string& description,
                 Fn fn) {
  m.def(name.c_str(), fn, description.c_str());
  Registry::instance().add(KernelInfo{name, chapter, variant, description});
}

}  // namespace ops_lab
