#include "bindings/registry.h"

#include <stdexcept>
#include <utility>

namespace ops_lab {

Registry& Registry::instance() {
  // C++11 起局部 static 的初始化是线程安全的，且只初始化一次。
  static Registry inst;
  return inst;
}

void Registry::add(KernelInfo info) {
  for (const auto& existing : entries_) {
    if (existing.name == info.name) {
      throw std::runtime_error(
          "算子名重复登记：" + info.name +
          "\n  同一个导出名被登记了两次。常见原因是从别的变体复制粘贴后忘了改名。");
    }
  }
  entries_.push_back(std::move(info));
}

}  // namespace ops_lab
