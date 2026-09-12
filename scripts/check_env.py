#!/usr/bin/env python3
"""环境体检 —— M1 的 S0 产出。

它不只是"打印版本号"。它的产出是 **S1 里 `_build.py` 要用的编译参数**，
所以参数与本模块共用同一个来源：``ops_lab.build_config``。

用法：

    python scripts/check_env.py            # 正常体检
    python scripts/check_env.py -v         # 附带完整编译/链接参数与剔除详情
    python scripts/check_env.py --json     # 机器可读输出
    python scripts/check_env.py --require-gpu   # 没有可见 GPU 就算失败（在有 GPU 的机器上用）

退出码：0 = 无致命问题；1 = 有致命问题（阻塞 S1）。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# --- 引导：先确定仓库位置，再剔除 PYTHONPATH 污染，最后才导入第三方库 ---
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python"))

from ops_lab.envguard import sanitize_syspath  # noqa: E402

_REMOVED = sanitize_syspath()

from ops_lab import build_config  # noqa: E402

OK, WARN, BAD = "OK", "WARN", "FAIL"


# ------------------------------------------------------------------ 输出小工具


class Report:
    """收集各分区的检查结果，最后统一打印与判定。"""

    def __init__(self) -> None:
        self.sections: list[tuple[str, list[tuple[str, str, str]]]] = []
        self.bad: list[str] = []
        self.warn: list[str] = []

    def add(self, title: str, rows: list[tuple[str, str, str]]) -> None:
        """rows: [(标签, 值, 状态)]，状态用于给值上色/标记。"""
        self.sections.append((title, rows))

    def render(self) -> None:
        for title, rows in self.sections:
            print(f"\n{title}")
            print("-" * 68)
            width = max((len(r[0]) for r in rows), default=0)
            for label, value, status in rows:
                mark = {OK: "  ", WARN: " !", BAD: " X"}.get(status, "  ")
                print(f"{mark} {label.ljust(width)}  {value}")


def _flag(status: str) -> str:
    return {OK: "  ", WARN: " !", BAD: " X"}.get(status, "  ")


# ------------------------------------------------------------------ 各分区体检


def check_python(cfg: build_config.BuildConfig, rep: Report) -> None:
    env_name = Path(cfg.prefix).name
    rows = [
        ("Python", cfg.python_version, OK),
        ("解释器", cfg.python_exe, OK),
        ("环境前缀", cfg.prefix, OK),
        (
            "环境类型",
            f"conda 环境 `{env_name}`" if cfg.is_conda else "非 conda 环境（venv/系统 Python）",
            OK if cfg.is_conda else WARN,
        ),
    ]
    if _REMOVED:
        rows.append(
            (
                "sys.path 净化",
                f"剔除了 {len(_REMOVED)} 个外来路径（PYTHONPATH 污染，加 -v 看详情）",
                WARN,
            )
        )
    rep.add("[1/7] Python 运行时", rows)


def check_torch(cfg: build_config.BuildConfig, rep: Report) -> None:
    rows = [
        ("torch", cfg.torch_version, OK),
        ("torch 的 CUDA", str(cfg.torch_cuda), OK),
        (
            "CXX11 ABI",
            f"{cfg.abi_cxx11}  →  编译需加 {cfg.abi_define}",
            OK,
        ),
        (
            "pybind11",
            f"{cfg.pybind11_version}（torch 自带，无需额外安装）"
            if cfg.pybind11_include
            else "缺失",
            OK if cfg.pybind11_include else BAD,
        ),
        ("torch include", str(cfg.include_dirs[0]) if cfg.include_dirs else "?", OK),
    ]

    # torch 可能在 C 层往 stderr 写过东西。已知的"找不到 CUDA runtime"提示我们
    # 已在 [5/7] 用更准确的方式报告（见 build_config.TORCH_CUDA_NOISE），
    # 但若出现意料之外的内容，必须暴露出来而不是吞掉。
    unexpected = [
        ln for ln in cfg.torch_stderr.splitlines()
        if ln.strip() and build_config.TORCH_CUDA_NOISE not in ln
    ]
    if unexpected:
        rows.append(("torch stderr", f"有 {len(unexpected)} 行非预期输出（加 -v 查看）", WARN))

    rep.add("[2/7] torch", rows)


def check_cuda(cfg: build_config.BuildConfig, rep: Report) -> None:
    archs = cfg.supported_archs
    arch_summary = (
        f"{archs[0]} … {archs[-1]}（共 {len(archs)} 个）" if archs else "无法获取"
    )
    rows = [
        ("nvcc", str(cfg.nvcc) if cfg.nvcc else "未找到", OK if cfg.nvcc else BAD),
        ("nvcc 版本", str(cfg.nvcc_version), OK if cfg.nvcc_version else BAD),
        ("CUDA_HOME", str(cfg.cuda_home) if cfg.cuda_home else "?", OK if cfg.cuda_home else BAD),
        ("nvcc 支持架构", arch_summary, OK if archs else WARN),
        ("目标架构", f"{cfg.arch}   ← 依据：{cfg.arch_source}", OK),
    ]
    rep.add("[3/7] CUDA 工具链", rows)


def check_host_tools(cfg: build_config.BuildConfig, rep: Report) -> None:
    rows = [
        ("C++ 编译器", f"{cfg.cxx}  {cfg.cxx_version or '?'}", OK if cfg.cxx_version else BAD),
        (
            "ninja",
            f"{cfg.ninja}" if cfg.ninja else "未找到（S1 的构建脚本必须有它）",
            OK if cfg.ninja else BAD,
        ),
        ("cmake", cfg.cmake or "未找到（仅 native/ 需要，可选）", OK if cfg.cmake else WARN),
    ]
    rep.add("[4/7] 宿主工具链与构建工具", rows)


def _gpu_rows() -> tuple[list[tuple[str, str, str]], dict]:
    """GPU 信息。沙箱里看不到设备是正常的，这里要优雅降级而不是崩。"""
    info: dict = {"visible": False}
    try:
        import torch
    except ImportError:
        return [("torch", "不可用，无法查询 GPU", BAD)], info

    if not torch.cuda.is_available():
        return (
            [
                ("可见 GPU", "0 个", WARN),
                (
                    "说明",
                    "torch.cuda.is_available() 为 False。在无 GPU 的沙箱/容器里属正常；"
                    "在有 GPU 的机器上，这里会显示设备名、算力与理论带宽。",
                    WARN,
                ),
                (
                    "如何确认本机 GPU",
                    "nvidia-smi  /  conda run -n <env> python -c "
                    "\"import torch;print(torch.cuda.get_device_name(0))\"",
                    OK,
                ),
            ],
            info,
        )

    prop = torch.cuda.get_device_properties(0)
    info.update(
        visible=True,
        name=prop.name,
        cc=f"{prop.major}.{prop.minor}",
        sm_count=prop.multi_processor_count,
        total_mem_gb=prop.total_memory / 1024**3,
    )
    rows = [
        ("设备", f"{prop.name}", OK),
        ("算力", f"{prop.major}.{prop.minor}  →  编译目标 sm_{prop.major}{prop.minor}", OK),
        ("SM 数量", str(prop.multi_processor_count), OK),
        ("显存", f"{info['total_mem_gb']:.1f} GiB", OK),
    ]

    # 理论带宽与 machine balance：用 nvidia-smi 拿显存频率与位宽。
    # 注意：权威实现是 kernels/common/device.h（运行期从 cudaDeviceProp 读）；
    # 这里是在**编译之前**跑的近似版本，两者都基于同一公式，不会漂移太多。
    bw = _theoretical_bandwidth_gbps()
    if bw:
        info["bandwidth_gbps"] = bw
        rows.append(("理论带宽", f"~{bw:.0f} GB/s（显存频率 × 位宽 / 8 × 2）", OK))
        sm_clock = _nvidia_smi_query("clocks.max.sm")
        if sm_clock:
            try:
                ghz = float(sm_clock) / 1000.0
                tflops = prop.multi_processor_count * 128 * 2 * ghz / 1000.0
                balance = tflops * 1e12 / (bw * 1e9)
                info.update(tflops_fp32=tflops, machine_balance=balance)
                rows.append(("FP32 算力", f"~{tflops:.1f} TFLOP/s（{prop.multi_processor_count} SM × 128 核 × 2 × {ghz:.2f} GHz）", OK))
                rows.append(
                    (
                        "Machine Balance",
                        f"~{balance:.0f} FLOP/Byte  →  算术强度低于它的算子都是 memory-bound",
                        OK,
                    )
                )
            except ValueError:
                pass
    return rows, info


def _nvidia_smi_query(field: str) -> str | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    val = proc.stdout.strip().splitlines()[0].strip() if proc.stdout.strip() else ""
    return val or None


def _theoretical_bandwidth_gbps() -> float | None:
    """显存频率(kHz) × 位宽(bit) / 8 × 2 → GB/s。"""
    clock_mhz = _nvidia_smi_query("clocks.max.memory")
    if not clock_mhz:
        return None
    width = None
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-q", "-d", "MEMORY"], capture_output=True, text=True, timeout=10
        )
        m = re.search(r"Bus Width\s*:\s*(\d+)\s*bit", proc.stdout)
        if m:
            width = int(m.group(1))
    except (OSError, subprocess.SubprocessError):
        pass
    if not width:
        return None
    try:
        clock_khz = float(clock_mhz) * 1000.0  # nvidia-smi 给的是 MHz
    except ValueError:
        return None
    return clock_khz * 1000.0 * (width / 8.0) * 2.0 / 1e9


def check_compile_contract(cfg: build_config.BuildConfig, rep: Report, verbose: bool) -> None:
    rows: list[tuple[str, str, str]] = [
        ("目标架构", cfg.arch_flag, OK),
        ("ABI 宏", cfg.abi_define, OK),
        ("include 目录", f"{len(cfg.include_dirs)} 个" + ("（-v 展开）" if not verbose else ""), OK),
        ("库目录", f"{len(cfg.library_dirs)} 个", OK),
        ("链接库", " ".join(f"-l{x}" for x in cfg.link_libs), OK),
    ]
    rep.add("[6/7] 编译参数契约（S1 的 _build.py 将使用这些）", rows)
    if verbose:
        # 走 stderr：保证 `--json` 时 stdout 只有纯 JSON
        out = sys.stderr
        print("\n  nvcc 编译参数：", file=out)
        for f in cfg.nvcc_compile_flags():
            print(f"    {f}", file=out)
        print("  g++ 编译参数：", file=out)
        for f in cfg.cxx_compile_flags():
            print(f"    {f}", file=out)
        print("  链接参数：", file=out)
        for f in cfg.link_flags():
            print(f"    {f}", file=out)


def check_diagnosis(cfg: build_config.BuildConfig, rep: Report, require_gpu: bool, gpu_ok: bool) -> None:
    rows: list[tuple[str, str, str]] = []
    for p in cfg.problems:
        rows.append(("致命", p, BAD))
    for w in cfg.warnings:
        rows.append(("警告", w, WARN))
    if require_gpu and not gpu_ok:
        rows.append(("致命", "--require-gpu 已指定，但没有可见的 GPU 设备", BAD))
    if not cfg.problems and not (require_gpu and not gpu_ok):
        rows.append(("结论", "未发现致命问题，可以进入 S1（构建最小闭环）", OK))
    rep.add("[7/7] 诊断结论", rows)


# ------------------------------------------------------------------------- 入口


def main() -> int:
    ap = argparse.ArgumentParser(description="cuda-ops-lab 环境体检")
    ap.add_argument("-v", "--verbose", action="store_true", help="打印完整编译/链接参数与被剔除的路径详情")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出（机器可读）")
    ap.add_argument("--require-gpu", action="store_true", help="没有可见 GPU 时判定为失败")
    args = ap.parse_args()

    if args.verbose and _REMOVED:
        print("[envguard] 被剔除的外来 sys.path 路径：", file=sys.stderr)
        for path, reason in _REMOVED:
            print(f"  - {path}\n    原因：{reason}", file=sys.stderr)

    cfg = build_config.detect()

    if args.verbose and cfg.torch_stderr.strip():
        print(
            "[torch] import 期间被捕获的 C 层 stderr 输出（默认已收掉，避免看起来像报错）：",
            file=sys.stderr,
        )
        for line in cfg.torch_stderr.splitlines():
            if line.strip():
                print(f"  {line}", file=sys.stderr)

    gpu_rows, gpu_info = _gpu_rows()

    rep = Report()
    check_python(cfg, rep)
    check_torch(cfg, rep)
    check_cuda(cfg, rep)
    check_host_tools(cfg, rep)
    rep.add("[5/7] GPU 与性能上限", gpu_rows)
    check_compile_contract(cfg, rep, args.verbose)
    check_diagnosis(cfg, rep, args.require_gpu, bool(gpu_info.get("visible")))

    if args.json:
        payload = {
            "python": {"version": cfg.python_version, "exe": cfg.python_exe, "prefix": cfg.prefix, "conda": cfg.is_conda},
            "torch": {"version": cfg.torch_version, "cuda": cfg.torch_cuda, "abi_cxx11": cfg.abi_cxx11},
            "cuda": {
                "nvcc": str(cfg.nvcc) if cfg.nvcc else None,
                "version": cfg.nvcc_version,
                "home": str(cfg.cuda_home) if cfg.cuda_home else None,
                "supported_archs": cfg.supported_archs,
            },
            "arch": {"target": cfg.arch, "source": cfg.arch_source, "flag": cfg.arch_flag},
            "tools": {"cxx": cfg.cxx, "ninja": str(cfg.ninja) if cfg.ninja else None, "cmake": cfg.cmake},
            "gpu": gpu_info,
            "flags": {
                "nvcc": cfg.nvcc_compile_flags(),
                "cxx": cfg.cxx_compile_flags(),
                "link": cfg.link_flags(),
            },
            "problems": cfg.problems,
            "warnings": cfg.warnings,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1 if cfg.has_fatal() else 0

    print("=" * 68)
    print(" cuda-ops-lab 环境体检")
    print("=" * 68)
    rep.render()
    print("\n" + "=" * 68)
    if cfg.has_fatal():
        print(" 结论：有致命问题，S1 无法开始（见上方标 X 的条目）")
        print("=" * 68)
        return 1
    print(" 结论：环境就绪")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
