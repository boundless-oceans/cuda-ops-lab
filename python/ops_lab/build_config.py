"""构建参数探测 —— `check_env.py` 与 `_build.py` 共用的唯一事实来源。

## 为什么要有这个模块

S0 的产出是"环境报告"，S1 的产出是"能编译的构建脚本"。如果两边各写一套参数，
报告就会变成装饰品：报告说 `-arch=sm_89`，构建脚本里却写着 `sm_86`，而且没人会发现。

所以参数只在这里算一次：

    build_config.detect()  ──┬──> scripts/check_env.py   打印给人看
                             └──> python/ops_lab/_build.py 生成 build.ninja

## 这里编码的都是实测结论

* torch 与 nvcc 的 CUDA 版本必须对齐（不一致会出现 c10::cuda 符号缺失）
* `_GLIBCXX_USE_CXX11_ABI` 必须跟 torch 一致（从 torch 自身读出后定义成 0 或 1）
* ninja 不一定在 PATH 上（可能在 `$CONDA_PREFIX/bin` 而不在 PATH）
* pybind11 用 torch 自带的那份，不额外安装
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["BuildConfig", "detect", "repo_root"]

REPO_ROOT = Path(__file__).resolve().parents[2]

# CUDA 版本 -> 官方支持的最高 GCC 主版本。
# 用错会得到一个非常难读的报错（"unsupported GNU version"），所以在探测阶段就拦。
_CUDA_MAX_GCC = {
    "11.0": 9, "11.1": 10, "11.2": 10, "11.3": 10, "11.4": 11,
    "11.5": 11, "11.6": 11, "11.7": 11, "11.8": 11,
    "12.0": 12, "12.1": 12, "12.2": 12, "12.3": 12, "12.4": 13,
}

_ARCH_RE = re.compile(r"(?:compute_|sm_)?(\d{2,3})")

# 项目的默认编译目标架构（compute capability 8.9，Ada 架构）。
# 这是**项目级常量**，不是对某台具体机器的假定：
#   * 能探测到 GPU 时，一律以实际设备的算力为准；
#   * 探测不到时可临时用环境变量 OPS_LAB_ARCH 覆盖；
#   * 只有在两者都不可用时才回退到这里。
# 若该架构不被当前 nvcc 支持，detect() 会报错而不是静默产出错误的目标码。
DEFAULT_ARCH = "sm_89"


def repo_root() -> Path:
    return REPO_ROOT


def _run(cmd: list[str], timeout: int = 30) -> str | None:
    """跑一条命令，成功返回合并后的输出，失败/超时返回 None（不抛异常）。"""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "") + (proc.stderr or "")


def _normalize_arch(text: str) -> str:
    """把 'compute_89' / 'sm_89' / '89' 统一成 'sm_89'。"""
    m = _ARCH_RE.search(text.strip())
    return f"sm_{m.group(1)}" if m else text.strip()


# 这个前缀的行是 torch 在"找不到 CUDA runtime"时的例行提示。
# 我们会在环境报告里用更准确的方式描述同一件事，所以默认不重复展示。
TORCH_CUDA_NOISE = "No CUDA runtime is found"


def _load_torch():
    """import torch，并把它在 C 层写进 stderr 的内容也一并捕获。

    为什么要动文件描述符：torch 找不到 CUDA runtime 时会从 **C++ 层**直接往
    stderr 打印 "No CUDA runtime is found, using CUDA_HOME=..."。
    这不是 Python 的 ``warnings``（实测 ``-W error::UserWarning`` 拦不住它），
    所以只能临时重定向 fd 2。

    返回 ``(torch, torch.utils.cpp_extension, 捕获到的 stderr 文本)``。
    """
    sys.stderr.flush()
    saved_fd = os.dup(2)
    with tempfile.TemporaryFile() as tmp:
        os.dup2(tmp.fileno(), 2)
        try:
            import torch
            from torch.utils import cpp_extension as ce
        finally:
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
        # 恢复 stderr 之后再读，避免把读取过程中的报错也吞进去
        tmp.seek(0)
        captured = tmp.read().decode("utf-8", "replace")
    return torch, ce, captured


# --------------------------------------------------------------------- 工具链探测


def _find_nvcc() -> Path | None:
    candidates: list[Path] = []
    if os.environ.get("CUDACXX"):
        candidates.append(Path(os.environ["CUDACXX"]))
    if shutil.which("nvcc"):
        candidates.append(Path(shutil.which("nvcc")))  # type: ignore[arg-type]
    candidates.append(Path("/usr/local/cuda/bin/nvcc"))
    # /usr/local/cuda-11.8/bin/nvcc 这类带版本号的目录，按版本号倒序
    for d in sorted(Path("/usr/local").glob("cuda-*"), reverse=True):
        candidates.append(d / "bin" / "nvcc")
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c.resolve()
    return None


def _nvcc_release(nvcc: Path) -> str | None:
    out = _run([str(nvcc), "--version"])
    if not out:
        return None
    m = re.search(r"release\s+([\d.]+)", out)
    if not m:
        return None
    parts = m.group(1).split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else m.group(1)


def _nvcc_supported_archs(nvcc: Path) -> list[str]:
    out = _run([str(nvcc), "--list-gpu-arch"])
    if not out:
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip().startswith("compute_")]


def _find_ninja() -> Path | None:
    candidates: list[Path] = []
    if os.environ.get("NINJA"):
        candidates.append(Path(os.environ["NINJA"]))
    if shutil.which("ninja"):
        candidates.append(Path(shutil.which("ninja")))  # type: ignore[arg-type]
    # 关键：ninja 经常装在 $CONDA_PREFIX/bin 但不在 PATH 上（沙箱实测如此）
    for prefix in (sys.prefix, os.environ.get("CONDA_PREFIX", "")):
        if prefix:
            candidates.append(Path(prefix) / "bin" / "ninja")
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c.resolve()
    return None


def _tool_version(exe: str) -> str | None:
    out = _run([exe, "--version"])
    if not out:
        return None
    return out.splitlines()[0].strip()


def _pybind11_info() -> tuple[Path | None, str | None]:
    """pybind11 用的是 torch 自带的那份（本仓库不额外安装）。"""
    try:
        import torch
    except ImportError:
        return None, None
    inc = Path(torch.__file__).parent / "include" / "pybind11"
    if not inc.is_dir():
        return None, None
    common = inc / "detail" / "common.h"
    version = None
    if common.is_file():
        try:
            text = common.read_text(encoding="utf-8", errors="ignore")
            parts = [
                re.search(rf"#define\s+PYBIND11_VERSION_{k}\s+(\d+)", text)
                for k in ("MAJOR", "MINOR", "PATCH")
            ]
            if all(parts):
                version = ".".join(m.group(1) for m in parts)  # type: ignore[union-attr]
        except OSError:
            pass
    return inc, version


def _detect_arch(nvcc: Path | None, supported: list[str]) -> tuple[str, str]:
    """决定 `-arch` 目标。优先级从"确定"到"猜"，并如实报告依据。"""
    env = os.environ.get("OPS_LAB_ARCH")
    if env:
        return _normalize_arch(env), "环境变量 OPS_LAB_ARCH 显式指定"

    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f"sm_{major}{minor}", "torch 报告的当前设备算力（最可靠）"
    except Exception:  # noqa: BLE001 - 探测失败不应该中断
        pass

    out = _run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    if out:
        raw = out.strip().splitlines()[0].strip().replace(".", "")
        if raw.isdigit():
            return f"sm_{raw}", "nvidia-smi 报告的当前设备算力"

    if f"compute_{DEFAULT_ARCH.replace('sm_', '')}" in supported:
        return DEFAULT_ARCH, (
            f"探测不到 GPU 且未设置 OPS_LAB_ARCH，回退到项目默认架构 {DEFAULT_ARCH}"
        )
    if supported:
        return _normalize_arch(supported[-1]), (
            f"探测不到 GPU，且项目默认架构 {DEFAULT_ARCH} 不被当前 nvcc 支持，"
            f"回退到其支持的最高架构"
        )
    return DEFAULT_ARCH, "探测不到 GPU 且 nvcc 无法列出架构，回退到项目默认架构"


# ------------------------------------------------------------------------- 结果


@dataclass
class BuildConfig:
    # --- Python / 环境 ---
    python_exe: str = ""
    python_version: str = ""
    prefix: str = ""
    is_conda: bool = False

    # --- torch ---
    torch_version: str = ""
    torch_cuda: str | None = None
    abi_cxx11: bool = True
    torch_stderr: str = ""

    # --- CUDA ---
    nvcc: Path | None = None
    nvcc_version: str | None = None
    cuda_home: Path | None = None
    cuda_include: Path | None = None
    cuda_lib: Path | None = None
    supported_archs: list[str] = field(default_factory=list)
    arch: str = ""
    arch_source: str = ""

    # --- 宿主编译器 / 构建工具 ---
    cxx: str = "g++"
    cxx_version: str | None = None
    ninja: Path | None = None
    cmake: str | None = None

    # --- pybind11 ---
    pybind11_include: Path | None = None
    pybind11_version: str | None = None

    # --- 编译/链接参数 ---
    include_dirs: list[Path] = field(default_factory=list)
    library_dirs: list[Path] = field(default_factory=list)
    link_libs: list[str] = field(default_factory=list)

    # --- 诊断 ---
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------- 参数生成

    @property
    def abi_define(self) -> str:
        return f"-D_GLIBCXX_USE_CXX11_ABI={1 if self.abi_cxx11 else 0}"

    @property
    def arch_flag(self) -> str:
        """生成 gencode。同时产出 PTX 便于未来用更新的驱动 JIT（这里只嵌 SASS+PTX）。"""
        num = self.arch.replace("sm_", "")
        return f"-gencode=arch=compute_{num},code=sm_{num}"

    def _common_defines(self) -> list[str]:
        return [
            self.abi_define,
            # 与 torch.utils.cpp_extension 保持一致：告诉 torch 头文件
            # "这是一个扩展模块"，并固定扩展名，避免符号可见性问题。
            "-DTORCH_API_INCLUDE_EXTENSION_H",
            "-DTORCH_EXTENSION_NAME=ops_lab_ext",
            # 把目标架构编译进去。运行期可以据此校验"扩展是按哪个架构编的"，
            # 排查"用错 arch"这类问题时不必去翻构建日志。
            f"-DOPS_LAB_TARGET_ARCH_NUM={self.arch.replace('sm_', '')}",
        ]

    def _includes(self) -> list[str]:
        return [f"-I{d}" for d in self.include_dirs]

    def _include_flags(self, for_nvcc: bool) -> list[str]:
        """仓库自己的头文件用 -I，第三方（torch / CUDA / python）用 -isystem。

        这样 `-Wall -Wextra` 只作用于我们的代码。否则 torch 与 CUDA 头文件里的
        历史遗留写法会刷出上百条警告，把真正的问题淹没。

        nvcc 不直接认 -isystem，需要经 -Xcompiler 转发给宿主编译器。
        """
        flags: list[str] = []
        for d in self.include_dirs:
            if d == REPO_ROOT:
                flags.append(f"-I{d}")
            elif for_nvcc:
                flags.append(f"-Xcompiler=-isystem,{d}")
            else:
                flags.append(f"-isystem{d}")
        return flags

    def nvcc_compile_flags(self) -> list[str]:
        return [
            "-std=c++17", "-O3",
            # 注意：nvcc **不认** -fPIC，必须经 -Xcompiler 转发给宿主编译器。
            # （实测：直接写 -fPIC 会得到 "nvcc fatal : Unknown option '-fPIC'"）
            "-Xcompiler=-fPIC",
            "-lineinfo",              # 给 ncu 用，代价很小
            self.arch_flag,
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-Xcompiler=-Wall,-Wextra",
            # ptxas 的资源报告（寄存器数 / spill / smem）。ninja 在成功时会丢弃
            # 命令输出，所以平时看不到；用 `_build.py -v` 就能看到，
            # S6 的 resource_report.py 也会解析这些字段。
            "-Xptxas=-v",
            *self._common_defines(),
            *self._include_flags(for_nvcc=True),
        ]

    def cxx_compile_flags(self) -> list[str]:
        return [
            "-std=c++17", "-O3", "-fPIC",
            "-Wall", "-Wextra",
            *self._common_defines(),
            *self._include_flags(for_nvcc=False),
        ]

    def link_flags(self) -> list[str]:
        """链接参数。

        注意：本仓库用 **nvcc** 链接（对象文件里有设备代码与 CUDA 运行期注册桩），
        而 nvcc **不认** gcc 的 `-Wl,...` 写法 —— 实测会报
        "nvcc fatal : Unknown option '-Wl,-rpath,...'"。

        也不能写成 `-Xcompiler=-Wl,-rpath,X`：`-Xcompiler` 的值是**逗号分隔**的，
        会被拆成 `-Wl` / `-rpath` / `X` 三个独立选项而失效。

        正确写法是 `-Xlinker -rpath -Xlinker <dir>`（每个链接器参数单独转发）。
        """
        flags = ["-shared"]
        flags += [f"-L{d}" for d in self.library_dirs]
        # 库必须排在对象文件之后（由调用方保证）：Ubuntu 默认 --as-needed，
        # 排在对象文件之前且当时无人引用的库会被直接丢掉。
        flags += [f"-l{lib}" for lib in self.link_libs]
        for d in self.library_dirs:
            flags += ["-Xlinker", "-rpath", "-Xlinker", str(d)]
        return flags

    def has_fatal(self) -> bool:
        return bool(self.problems)


# ------------------------------------------------------------------------- 主入口


def detect() -> BuildConfig:
    cfg = BuildConfig()

    # ---- Python 侧 ----
    cfg.python_exe = sys.executable
    cfg.python_version = ".".join(str(v) for v in sys.version_info[:3])
    cfg.prefix = sys.prefix
    cfg.is_conda = Path(sys.prefix, "conda-meta").is_dir()

    # ---- torch ----
    try:
        torch, ce, captured = _load_torch()

        cfg.torch_stderr = captured
        cfg.torch_version = torch.__version__
        cfg.torch_cuda = torch.version.cuda
        cfg.abi_cxx11 = bool(torch._C._GLIBCXX_USE_CXX11_ABI)
        cfg.include_dirs += [Path(p) for p in ce.include_paths()]
        cfg.library_dirs += [Path(p) for p in ce.library_paths()]
    except ImportError as exc:
        cfg.problems.append(
            f"无法 import torch：{exc}。请先激活正确的 conda 环境"
            "（例如 conda activate <本仓库的环境名>）"
        )
        return cfg

    # ---- nvcc ----
    cfg.nvcc = _find_nvcc()
    if cfg.nvcc is None:
        cfg.problems.append(
            "找不到 nvcc。编译 CUDA 扩展必须有 CUDA Toolkit（不只是驱动）。"
            "检查 /usr/local/cuda-*/bin/nvcc 是否存在。"
        )
    else:
        cfg.cuda_home = cfg.nvcc.parent.parent
        cfg.cuda_include = cfg.cuda_home / "include"
        cfg.cuda_lib = cfg.cuda_home / "lib64"
        cfg.nvcc_version = _nvcc_release(cfg.nvcc)
        cfg.supported_archs = _nvcc_supported_archs(cfg.nvcc)

        if not cfg.cuda_include.is_dir():
            cfg.problems.append(f"CUDA 头文件目录不存在：{cfg.cuda_include}")
        else:
            cfg.include_dirs.append(cfg.cuda_include)
        if not cfg.cuda_lib.is_dir():
            cfg.problems.append(f"CUDA 库目录不存在：{cfg.cuda_lib}")
        else:
            cfg.library_dirs.append(cfg.cuda_lib)

        # torch 与 nvcc 的 CUDA 版本必须一致
        if cfg.torch_cuda and cfg.nvcc_version and cfg.torch_cuda != cfg.nvcc_version:
            cfg.problems.append(
                f"CUDA 版本不匹配：torch 编译时用的是 CUDA {cfg.torch_cuda}，"
                f"而 nvcc 是 {cfg.nvcc_version}。二者必须一致，否则会出现 "
                f"c10::cuda 符号缺失之类的诡异链接错误。"
            )

    cfg.arch, cfg.arch_source = _detect_arch(cfg.nvcc, cfg.supported_archs)
    if cfg.arch not in [f"sm_{a.split('_')[1]}" for a in cfg.supported_archs] and cfg.supported_archs:
        cfg.problems.append(
            f"目标架构 {cfg.arch} 不被本机 nvcc 支持。"
            f"支持的架构：{', '.join(cfg.supported_archs)}"
        )

    # ---- 宿主编译器 ----
    cfg.cxx = os.environ.get("CXX") or "g++"
    if shutil.which(cfg.cxx):
        cfg.cxx_version = _tool_version(cfg.cxx)
        if cfg.nvcc_version and cfg.cxx_version:
            m = re.search(r"(\d+)", cfg.cxx_version)
            gcc_major = int(m.group(1)) if m else None
            max_gcc = _CUDA_MAX_GCC.get(cfg.nvcc_version)
            if gcc_major and max_gcc and gcc_major > max_gcc:
                cfg.problems.append(
                    f"{cfg.cxx} 主版本 {gcc_major} 超出 CUDA {cfg.nvcc_version} 支持的上限 "
                    f"（GCC {max_gcc}）。nvcc 会报 'unsupported GNU version'。"
                )
    else:
        cfg.problems.append(f"找不到宿主编译器 {cfg.cxx}")

    # ---- 构建工具 ----
    cfg.ninja = _find_ninja()
    if cfg.ninja is None:
        cfg.problems.append(
            "找不到 ninja。本仓库用自研 ninja 构建脚本，必须有 ninja 可执行文件。"
            "注意：conda 的 ninja 包不含 Python 模块，所以 "
            "torch.utils.cpp_extension.load() 那条路走不通，只能靠可执行文件。"
        )
    cfg.cmake = shutil.which("cmake")

    # ---- pybind11（用 torch 自带）----
    cfg.pybind11_include, cfg.pybind11_version = _pybind11_info()
    if cfg.pybind11_include is None:
        cfg.problems.append(
            "torch/include/pybind11 不存在。本仓库用 torch 自带的 pybind11，"
            "不额外安装；若缺失说明 torch 安装不完整。"
        )
    else:
        # torch/include 已经在 include_dirs 里，pybind11 就在它下面，
        # 所以头文件用 #include <pybind11/pybind11.h> 即可，无需再加 -I
        pass

    # ---- python 头文件 ----
    py_inc = Path(sys.prefix) / "include" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    if py_inc.is_dir():
        cfg.include_dirs.append(py_inc)
    else:
        cfg.problems.append(f"找不到 Python 头文件目录：{py_inc}")

    # ---- 仓库自己的头文件根（kernels/ 用仓库根做 include 前缀）----
    if REPO_ROOT.is_dir():
        cfg.include_dirs.append(REPO_ROOT)

    # ---- 链接库 ----
    cfg.link_libs = ["torch_python", "torch", "torch_cuda", "torch_cpu", "c10_cuda", "c10"]
    if cfg.cuda_lib and cfg.cuda_lib.is_dir():
        cfg.link_libs.append("cudart")

    return cfg
