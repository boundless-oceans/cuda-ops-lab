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

## 当前阶段（S1-1）

只把 `kernels/**/*.cu` 编译成对象文件，产物在 `build/ext/obj/`。
链接成 Python 扩展是 S1-2 引入绑定之后的事。

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


def _nja(path: Path) -> str:
    """ninja 路径转义。

    本项目路径里没有空格和冒号，但转义是无害的保险：
    ninja 用空格分隔 token、用冒号分隔 build 语句的输入输出。
    """
    return str(path).replace("$", "$$").replace(" ", "$ ").replace(":", "$:")


def collect_kernel_sources() -> list[Path]:
    """kernels/ 下所有 .cu，排序后返回（保证生成的构建文件可复现）。"""
    return sorted((_REPO_ROOT / "kernels").rglob("*.cu"))


def object_path(src: Path) -> Path:
    """对象文件路径：镜像源码目录结构，便于和源文件对照。"""
    rel = src.relative_to(_REPO_ROOT)
    return BUILD_DIR / OBJ_SUBDIR / rel.parent / (rel.name + ".o")


def generate_ninja(cfg: build_config.BuildConfig, sources: list[Path]) -> Path:
    """生成 build.ninja。每次都重新生成（很便宜），增量性交给 ninja。"""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    ninja_file = BUILD_DIR / NINJA_FILE_NAME

    flags = " ".join(cfg.nvcc_compile_flags())
    lines: list[str] = [
        "# 本文件由 python/ops_lab/_build.py 自动生成，请勿手工编辑。",
        "# 要改编译参数请改 ops_lab/build_config.py，要改构建逻辑请改 _build.py。",
        "",
        "ninja_required_version = 1.7",
        "",
        f"nvcc = {_nja(cfg.nvcc) if cfg.nvcc else 'nvcc'}",
        f"nvccflags = {flags}",
        "",
        "rule nvcc_compile",
        "  command = $nvcc $nvccflags -MMD -MF ${out}.d -c $in -o $out",
        "  depfile = ${out}.d",
        "  deps = gcc",
        "  description = NVCC $in",
        "",
    ]

    outputs: list[str] = []
    for src in sources:
        obj = object_path(src)
        outputs.append(_nja(obj))
        lines.append(f"build {_nja(obj)}: nvcc_compile {_nja(src)}")
    lines.append("")

    # 对象文件全部建好即完成（S1-1 没有链接目标）。
    # ninja 会自动创建输出文件所在目录。
    lines.append("default " + " ".join(outputs))
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

    sources = collect_kernel_sources()
    if not sources:
        print("没有找到任何 kernels/**/*.cu", file=sys.stderr)
        return 1

    ninja_file = generate_ninja(cfg, sources)

    print(f"[_build] 仓库根    {_REPO_ROOT}")
    print(f"[_build] 构建目录  {BUILD_DIR}")
    print(f"[_build] 目标架构  {cfg.arch}（{cfg.arch_source}）")
    print(f"[_build] 源文件    {len(sources)} 个 .cu")

    code = run_ninja(cfg, ninja_file, args.verbose)
    if code != 0:
        return code

    print("[_build] 构建完成")
    print(f"[_build] 产物：{len(sources)} 个对象文件，位于 {BUILD_DIR / OBJ_SUBDIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
