"""设备信息查询。

## 两条数据来源，优先用扩展

1. **扩展里的 `device_info()`**（`kernels/common/device.h`，读 `cudaDeviceProp`）——
   **权威来源**，`native/` 的纯 CUDA 可执行文件用的是同一份，两边不会给出不同的
   "理论峰值"。需要扩展已构建。
2. **nvidia-smi** —— 构建之前也能用，所以 `check_env.py` 依赖它兜底。

之所以必须先试扩展：**显存位宽在 nvidia-smi 上拿不到**。实测驱动 580.178.04
（CUDA 13.0）的 `nvidia-smi -q -d MEMORY` 只输出 FB / BAR1 / Conf Compute 三段，
没有 "Bus Width"。而算峰值带宽必须有位宽，只能用 `cudaDeviceProp.memoryBusWidth`。

## 查不到就返回 None，不抛异常

沙箱和 CI 里没有 GPU，两条路都会失败。这种环境应当**优雅降级**
（把峰值带宽显示成"未知"），而不是让整个脚本崩掉。
"""

from __future__ import annotations

import re
import subprocess

__all__ = [
    "device_info",
    "nvidia_smi_query",
    "device_name",
    "memory_clock_mhz",
    "memory_bus_width_bits",
    "theoretical_bandwidth_gbps",
]


def device_info() -> dict | None:
    """从扩展里读设备信息（含位宽与理论峰值带宽）。拿不到返回 None。

    注意：**不会触发构建**。`check_env.py` 的职责就是在构建之前跑，
    如果这里顺手触发了自动构建，那个脚本就自我否定了。
    """
    try:
        from . import _extension

        if not _extension.is_built():
            return None
        return _extension.get().device_info()
    except Exception:  # noqa: BLE001 - 扩展缺失/过期/无设备，一律降级
        return None


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
    from_ext = device_info()
    if from_ext:
        return from_ext.get("name")
    return nvidia_smi_query("name")


def memory_clock_mhz() -> float | None:
    from_ext = device_info()
    if from_ext and from_ext.get("mem_clock_khz"):
        return float(from_ext["mem_clock_khz"]) / 1000.0  # cudaDeviceProp 用 kHz

    raw = nvidia_smi_query("clocks.max.memory")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def memory_bus_width_bits() -> int | None:
    """显存位宽（bit）。

    **优先从扩展读** `cudaDeviceProp.memoryBusWidth`。

    nvidia-smi 这条路只是兜底：`--query-gpu` 没有位宽字段，只能在 `-q -d MEMORY`
    的文本里找 "Bus Width" —— 而多数驱动版本**根本不输出这一行**（实测
    driver 580 就是这样）。所以这个兜底大概率返回 None，属于预期行为。
    """
    from_ext = device_info()
    if from_ext and from_ext.get("mem_bus_width"):
        return int(from_ext["mem_bus_width"])

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
    from_ext = device_info()
    if from_ext and from_ext.get("mem_bandwidth_gbps"):
        return float(from_ext["mem_bandwidth_gbps"])

    clock_mhz = memory_clock_mhz()
    width_bits = memory_bus_width_bits()
    if not clock_mhz or not width_bits:
        return None
    return (clock_mhz * 1e6) * (width_bits / 8.0) * 2.0 / 1e9
