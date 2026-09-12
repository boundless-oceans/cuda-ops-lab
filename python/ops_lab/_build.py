#!/usr/bin/env python3
"""自研 ninja 构建（M1 / S1）。

## 为什么不用 torch.utils.cpp_extension

三个实测原因，不是洁癖：

1. 环境里的 conda `ninja` 包**不含 Python 模块**，只有可执行文件，
   而 `cpp_extension` 需要 `import ninja`；
2. ninja 可执行文件不一定在 `PATH` 上（可能只存在于 `$CONDA_PREFIX/bin`）；
3. 自研才能完全控制命令行 —— 把真实参数打印出来、按需开启 `-Xptxas=-v`，
   而 `cpp_extension` 把这一切藏在黑盒里。

## 参数来源

这里**不硬编码任何编译参数**，全部来自 `ops_lab.build_config.detect()`，
与 `scripts/check_env.py` 用的是同一份来源（见 design §M1.1 的 S0）。
这样"报告里写的参数"和"真正编译用的参数"不可能漂移。

## 构建产物

    build/ext/obj/kernels/**/*.cu.o     由 nvcc 编译
    build/ext/obj/bindings/**/*.cpp.o   由 g++ 编译
    build/ext/ops_lab_ext.so            由 nvcc 链接（存在 bindings/ 时）

如果 `bindings/` 为空，则只编译对象文件、不链接 —— 这仍然是有用的模式
（验证工具链、产出 ptxas 报告），不是临时状态。

## 为什么要用 nvcc 而不是 g++ 链接

对象文件里含设备代码（fatbin）与 CUDA 运行期注册桩，用 nvcc 链接能自动带上
正确的运行期支持。库排在对象文件之后是链接器的常规顺序，照做即可。

（实测补充：本机 nvcc 与 g++ 都**没有**启用 `--as-needed`，因此未被引用的
`-ltorch` 也会被记进 `DT_NEEDED`。副作用是 S1-2 阶段即使不写 torch 代码，
rpath 也已经被真实验证过了。）

用法：

    python python/ops_lab/_build.py            # 增量构建
    python python/ops_lab/_build.py -v         # 打印真实命令行与 ptxas 资源报告
    python python/ops_lab/_build.py --clean    # 清空 build/ 后重编
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# --- 引导：定位仓库根 → 把 python/ 放进 sys.path → 再 import ops_lab ---
# 仓库根只能在这里算一次（import 之后就没机会了）。
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))

from ops_lab import build_config  # noqa: E402
from ops_lab.envguard import sanitize_syspath  # noqa: E402

BUILD_DIR = _REPO_ROOT / "build" / "ext"
OBJ_SUBDIR = "obj"
NINJA_FILE_NAME = "build.ninja"
EXTENSION_NAME = "ops_lab_ext"
EXTENSION_SUFFIX = ".so"


def _nja(path: Path) -> str:
    """ninja 路径转义。

    本项目路径里没有空格和冒号，但转义是无害的保险：
    ninja 用空格分隔 token、用冒号分隔 build 语句的输入输出。
    """
    return str(path).replace("$", "$$").replace(" ", "$ ").replace(":", "$:")


def collect_kernel_sources() -> list[Path]:
    """kernels/ 下所有 .cu，排序后返回（保证生成的构建文件可复现）。"""
    return sorted((_REPO_ROOT / "kernels").rglob("*.cu"))


def collect_binding_sources() -> list[Path]:
    """bindings/ 下所有 .cpp（pybind11 胶水层）。"""
    return sorted((_REPO_ROOT / "bindings").rglob("*.cpp"))


def object_path(src: Path) -> Path:
    """对象文件路径：镜像源码目录结构，便于和源文件对照。"""
    rel = src.relative_to(_REPO_ROOT)
    return BUILD_DIR / OBJ_SUBDIR / rel.parent / (rel.name + ".o")


def extension_path() -> Path:
    return BUILD_DIR / (EXTENSION_NAME + EXTENSION_SUFFIX)


def generate_ninja(cfg: build_config.BuildConfig,
                   kernels: list[Path],
                   bindings: list[Path]) -> Path:
    """生成 build.ninja。每次都重新生成（很便宜），增量性交给 ninja。"""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    ninja_file = BUILD_DIR / NINJA_FILE_NAME

    lines: list[str] = [
        "# 本文件由 python/ops_lab/_build.py 自动生成，请勿手工编辑。",
        "# 要改编译参数请改 ops_lab/build_config.py，要改构建逻辑请改 _build.py。",
        "",
        "ninja_required_version = 1.7",
        "",
        f"nvcc = {_nja(cfg.nvcc) if cfg.nvcc else 'nvcc'}",
        f"nvccflags = {' '.join(cfg.nvcc_compile_flags())}",
        f"cxx = {cfg.cxx}",
        f"cxxflags = {' '.join(cfg.cxx_compile_flags())}",
        f"linkflags = {' '.join(cfg.link_flags())}",
        "",
        "rule nvcc_compile",
        "  command = $nvcc $nvccflags -MMD -MF ${out}.d -c $in -o $out",
        "  depfile = ${out}.d",
        "  deps = gcc",
        "  description = NVCC $in",
        "",
        "rule cxx_compile",
        "  command = $cxx $cxxflags -MMD -MF ${out}.d -c $in -o $out",
        "  depfile = ${out}.d",
        "  deps = gcc",
        "  description = CXX  $in",
        "",
        "rule link",
        # 库（在 $linkflags 里）排在 $in 之后是链接器的常规顺序。
        "  command = $nvcc $in $linkflags -o $out",
        "  description = LINK $out",
        "",
    ]

    objects: list[str] = []
    for src in kernels:
        obj = object_path(src)
        objects.append(_nja(obj))
        lines.append(f"build {_nja(obj)}: nvcc_compile {_nja(src)}")
    for src in bindings:
        obj = object_path(src)
        objects.append(_nja(obj))
        lines.append(f"build {_nja(obj)}: cxx_compile {_nja(src)}")
    lines.append("")

    if bindings:
        so = extension_path()
        lines.append(f"build {_nja(so)}: link {' '.join(objects)}")
        lines.append("")
        lines.append(f"default {_nja(so)}")
    else:
        # 没有绑定源码时不链接，只产出对象文件
        lines.append("default " + " ".join(objects))
    lines.append("")

    ninja_file.write_text("\n".join(lines), encoding="utf-8")
    return ninja_file


def run_ninja(cfg: build_config.BuildConfig, ninja_file: Path, verbose: bool) -> int:
    """调用 ninja。返回退出码（0 表示成功）。"""
    if cfg.ninja is None:
        print(
            "找不到 ninja 可执行文件，无法构建。\n"
            "先运行 `python scripts/check_env.py` 查看诊断。",
            file=sys.stderr,
        )
        return 1

    # -C 让 ninja 先切到构建目录（.ninja_log / .ninja_deps 也就落在那里，不污染仓库根）
    cmd = [str(cfg.ninja), "-C", str(BUILD_DIR), "-f", ninja_file.name]
    if verbose:
        cmd.append("-v")

    print(f"[_build] 执行：{' '.join(cmd)}")
    # ninja 是子进程，直接写 fd；不先把 Python 侧的缓冲刷出去，
    # 输出顺序会错乱（stdout 被缓冲、stderr 不会）。
    sys.stdout.flush()

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        # ninja 已经把真正的错误打印出来了，这里只补一行结论，
        # 不再甩一个 Python traceback 淹没它。
        print(f"\n[_build] 构建失败（ninja 退出码 {exc.returncode}）", file=sys.stderr)
        return exc.returncode
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="cuda-ops-lab 扩展构建")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="打印真实命令行与 ptxas 资源报告（默认只在失败时显示）")
    ap.add_argument("--clean", action="store_true", help="删除 build/ 后重新构建")
    args = ap.parse_args()

    sanitize_syspath(verbose=args.verbose)

    if args.clean:
        shutil.rmtree(BUILD_DIR, ignore_errors=True)

    cfg = build_config.detect()
    if cfg.has_fatal():
        print("环境有致命问题，拒绝构建：", file=sys.stderr)
        for problem in cfg.problems:
            print(f"  X {problem}", file=sys.stderr)
        print("\n先运行：python scripts/check_env.py", file=sys.stderr)
        return 1

    kernels = collect_kernel_sources()
    bindings = collect_binding_sources()
    if not kernels and not bindings:
        print("没有找到任何 kernels/**/*.cu 或 bindings/**/*.cpp", file=sys.stderr)
        return 1

    ninja_file = generate_ninja(cfg, kernels, bindings)

    print(f"[_build] 仓库根    {_REPO_ROOT}")
    print(f"[_build] 构建目录  {BUILD_DIR}")
    print(f"[_build] 目标架构  {cfg.arch}（{cfg.arch_source}）")
    print(f"[_build] 源文件    {len(kernels)} 个 .cu，{len(bindings)} 个 .cpp")

    code = run_ninja(cfg, ninja_file, args.verbose)
    if code != 0:
        return code

    print("[_build] 构建完成")
    if bindings:
        print(f"[_build] 产物：{extension_path()}")
    else:
        print(f"[_build] 产物：{len(kernels)} 个对象文件（无 bindings/，未链接）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
