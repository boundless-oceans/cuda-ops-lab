"""设备信息查询（纯 Python，**构建之前**就能用）。

## 为什么用 nvidia-smi 而不是 C++ 的 device.h

`scripts/check_env.py` 必须在构建之前运行 —— 那时扩展还不存在，用不了
`kernels/common/device.h`。所以这里用 nvidia-smi 作为"构建前的近似"。

**权威实现仍然是 `kernels/common/device.h`**（直接读 `cudaDeviceProp`，`native/`
也用它）。两者用同一个公式，差别只在数据来源：nvidia-smi 的显存时钟单位是 MHz，
`cudaDeviceProp.memoryClockRate` 是 kHz。

之所以把它单独成一个模块，是因为 `bench/run_bench.py` 也要算 %峰值 ——
同一份公式不该有两处手写。

## 查不到就返回 None，不抛异常

沙箱和 CI 里没有 GPU，`nvidia-smi` 会直接失败。这种环境应当**优雅降级**
（把峰值带宽显示成"未知"），而不是让整个脚本崩掉。
"""

from __future__ import annotations

import re
import subprocess

__all__ = [
    "nvidia_smi_query",
    "device_name",
    "memory_clock_mhz",
    "memory_bus_width_bits",
    "theoretical_bandwidth_gbps",
]


def nvidia_smi_query(field: str) -> str | None:
    """查一个 nvidia-smi 字段；查不到返回 None（无 GPU、无驱动都会走到这里）。"""
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    lines = proc.stdout.strip().splitlines()
    return lines[0].strip() if lines else None


def device_name() -> str | None:
    return nvidia_smi_query("name")


def memory_clock_mhz() -> float | None:
    raw = nvidia_smi_query("clocks.max.memory")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def memory_bus_width_bits() -> int | None:
    """显存位宽。

    nvidia-smi 的 `--query-gpu` 没有位宽字段，只能在 `-q -d MEMORY` 的详细
    输出里解析文本 —— 所以这个函数比其它的脆弱，解析失败时返回 None。
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-q", "-d", "MEMORY"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"Bus Width\s*:\s*(\d+)\s*bit", proc.stdout)
    return int(match.group(1)) if match else None


def theoretical_bandwidth_gbps() -> float | None:
    """理论峰值带宽 = 显存时钟(Hz) × 位宽(bit) / 8 × 2（DDR 双边沿）。

    这是**保守**估计：厂商手册的标称带宽往往更高，因为用的是颗粒的等效数据率
    （MT/s）而不是实际时钟。对"判断离上限多远"这件事，宁可低估峰值，
    也不要高估自己 —— 高估会让你以为已经到顶了，其实还有空间。
    """
    clock_mhz = memory_clock_mhz()
    width_bits = memory_bus_width_bits()
    if not clock_mhz or not width_bits:
        return None
    return (clock_mhz * 1e6) * (width_bits / 8.0) * 2.0 / 1e9
