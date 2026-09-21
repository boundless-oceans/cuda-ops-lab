# native/CMakeLists.txt
#
# 纯 CUDA 可执行文件（不依赖 Python / torch）。
#
# 用法：
#     cmake -B build/native -S native
#     cmake --build build/native -j
#     ./build/native/main_02_elementwise
#     ncu --set full ./build/native/main_02_elementwise 16777216 10
#
# ## 为什么这里用 CMake，而 Python 扩展用自研 ninja 脚本
#
# 产物形态不同：这里是**可执行文件**，要的正是"能独立构建、能直接丢给 ncu"；
# 而扩展那边必须与 torch 的 include/lib 精确对接，自研脚本更好控制。
# 两条线各自独立，互不依赖 —— 这也符合本仓库的分层原则。
#
# ## kernel 源码是复用的，不是另抄一份
#
# 下面直接 GLOB `kernels/` 下的全部 .cu。所以这条线是**真对照**：
# 同一个 kernel、同一份 benchmark.h / device.h，只是外面没有 Python。
# 哪天有人往 kernels/ 里塞了 torch 符号，这里会第一个编译失败。
cmake_minimum_required(VERSION 3.20)

project(cuda_ops_lab_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 目标架构。
#
# 注意不能写成 `if(NOT CMAKE_CUDA_ARCHITECTURES)`：CMake 在 project() 阶段就会把
# 这个变量填成**编译器默认值**（CUDA 11.8 是 52），于是判断永远不成立，
# 最终编出 sm_52 的 cubin —— 在 sm_89 上会因为"no kernel image available"起不来。
# 所以用一个自己的变量，最后无条件覆盖。优先级：
#   -DOPS_LAB_CUDA_ARCH=<NN>  >  环境变量 OPS_LAB_ARCH  >  项目默认 89
set(OPS_LAB_CUDA_ARCH "" CACHE STRING "CUDA 目标架构编号（如 89）；留空则自动决定")
if(NOT OPS_LAB_CUDA_ARCH)
  if(DEFINED ENV{OPS_LAB_ARCH})
    string(REGEX REPLACE "^sm_" "" OPS_LAB_CUDA_ARCH "$ENV{OPS_LAB_ARCH}")
  else()
    set(OPS_LAB_CUDA_ARCH 89)
  endif()
endif()
set(CMAKE_CUDA_ARCHITECTURES "${OPS_LAB_CUDA_ARCH}")
message(STATUS "CUDA 目标架构: sm_${CMAKE_CUDA_ARCHITECTURES}")

# 复用全部 kernel 源码
file(GLOB_RECURSE OPS_LAB_KERNELS CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/*.cu")
list(LENGTH OPS_LAB_KERNELS _n_kernels)
message(STATUS "复用的 kernel 源文件: ${_n_kernels} 个")

function(ops_lab_add_native target main_source)
  add_executable(${target} ${main_source} ${OPS_LAB_KERNELS})

  # kernels/ 用仓库根做 include 前缀（例如 "kernels/common/device.h"）
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")

  # 只对宿主机代码开警告；nvcc 自己的告警另算
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)

  # 设备代码在宿主编译器看来是 -fPIC 无关的，这里不需要 PIC
  set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endfunction()

ops_lab_add_native(main_01_execution main_01_execution.cu)
ops_lab_add_native(main_02_elementwise main_02_elementwise.cu)
ops_lab_add_native(main_03_reduction main_03_reduction.cu)
