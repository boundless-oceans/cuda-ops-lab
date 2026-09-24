"""GPU 时钟状态 —— 以及为什么基准必须先"预热到稳态"。

## 这个模块是被一个 14% 的假差异逼出来的

第 3 章的实测出现过自相矛盾的现象：同一个 kernel、同样的 67.1 MB，
native 线报 **313.3 µs（83.6% 峰值）**，Python 线报 **275.5 µs（95.2%）**。
两条线用的是同一份 kernel、同一套计时公式。差在哪？

不是 kernel，是**显存时钟**：

    GPU 状态（开跑前）        SM 1890 MHz，显存 **7001 MHz**，41°C， 9.85 W
    GPU 状态（1.5 s 负载后）  SM 2325 MHz，显存 **8001 MHz**，47°C，55.21 W

    8001 / 7001 = 1.143      313.3 / 275.5 = 1.137

把两边的**真实利用率**算出来就一目了然（理论峰值是按额定 8001 MHz 算的）：

    冷态  313.3 µs → 214.2 GB/s，而那一档时钟的峰值是 224.0 GB/s  → **95.6%**
    稳态  274.0 µs → 244.9 GB/s，峰值 256.0 GB/s                  → **95.7%**

**两个都是 95.6%，只是分母不同。** 冷机跑出来的那几次测量并不慢，
是除法的分母用了它当时达不到的时钟。

## 所以基准脚本必须做两件事

1. **测量之前把 GPU 推到时钟稳态**（`steady_state`），否则最先测的几个变体
   会被凭空打上"只有 83% 峰值"的标签。
2. **把采样到的时钟写进结果表头**（`describe`）。读者必须能一眼判断
   这一列数字可不可信 —— 否则这个坑还会有人再踩一次。

## 它**不能**解释的：第 1 章那个 `probe_smem48k`

一度以为这台机器的时钟爬升顺便解释了第 1 章那个悬案（`probe_smem48k` 比
`probe_smem0` 快 5%）—— **那个归因是错的**。用预热 + 交错重测之后它照样复现
（238.7 vs 226.4 GB/s，轮间极差只有 0.5 µs），说明那是一个**真实的硬件效应**，
不是测量artifact。教训：时钟能解释"整张表都偏低"，解释不了"表内某一个变体偏高"。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time

__all__ = [
    "query_state",
    "parse_state",
    "at_rated_memory_clock",
    "steady_state",
    "describe",
    "describe_samples",
    "samples_at_rated",
]

# 一次查询同时拿到"当前值"和"额定上限"。分两次查会有个隐患：
# 两次 subprocess 之间时钟可能已经变了，于是"当前值"和"上限"来自不同时刻。
_SMI_FIELDS = "clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,temperature.gpu,power.draw"

_FIELDS = (
    "sm_clock_mhz",
    "mem_clock_mhz",
    "max_sm_clock_mhz",
    "max_mem_clock_mhz",
    "temperature_c",
    "power_w",
)

_MHZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*MHz")
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")


def parse_state(csv_line: str) -> dict:
    """把 nvidia-smi 的一行 CSV 解析成字典。**纯函数，离线可测。**

    字段顺序与 `_SMI_FIELDS` 一致。任何一项解析不出来就是 None（例如
    笔记本上 `power.draw` 常常是 `[N/A]`）—— **不抛异常**：采样失败不该
    把整个基准带崩，只是少一列信息。
    """
    parts = [p.strip() for p in csv_line.strip().split(",")]
    # 先把所有键填成 None：字段比预期少时（nvidia-smi 版本差异）也要能安全取值，
    # 而不是让调用方撞 KeyError
    out: dict = {key: None for key in _FIELDS}
    for key, raw in zip(_FIELDS, parts):
        if key.endswith("_mhz"):
            m = _MHZ_RE.search(raw)
        else:
            m = _NUM_RE.search(raw)
        out[key] = float(m.group(1)) if m else None
    return out


def query_state() -> dict | None:
    """采样一次 GPU 状态。nvidia-smi 不存在或失败时返回 None（不抛异常）。"""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_SMI_FIELDS}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return parse_state(proc.stdout.strip().splitlines()[0])


def at_rated_memory_clock(state: dict, tolerance: float = 0.01) -> bool | None:
    """显存时钟是否已经到额定值。拿不到时钟信息时返回 None（而不是猜）。"""
    cur = state.get("mem_clock_mhz")
    rated = state.get("max_mem_clock_mhz")
    if cur is None or rated is None or rated <= 0:
        return None
    return cur >= rated * (1.0 - tolerance)


def _apply_load(load, seconds: float) -> None:
    """连续施加负载 seconds 秒 —— **中途一次都不查询**。

    为什么必须强调"中途不查询"：查询要走一次 `nvidia-smi` 子进程（几十毫秒）
    再加一个 sleep。如果每跑一次 kernel 就查一次，GPU 的**占空比会掉到千分之一**，
    时钟根本爬不上去 —— 第一版就是这么写的，实测结果是"预热 8 秒，时钟一动不动"
    （表头采样到 SM 1890 MHz / 9 W，一台空转的机器）。

    改成"先压一段时间、再查"之后，1.5 秒就能从 7001 爬到 8001 MHz。
    """
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        load()


def steady_state(
    load,
    *,
    min_s: float = 1.5,
    burst_s: float = 0.5,
    budget_s: float = 10.0,
    tolerance: float = 0.01,
    fallback_s: float = 2.0,
) -> dict:
    """施加负载直到显存时钟爬到额定值（或超时），返回最后一次采样。

    节奏是"**连续压 min_s 秒 → 查一次 → 没到额定就再压 burst_s 秒 → 再查**"，
    直到到顶或超出 budget_s。`load` 是无参可调用对象，一次调用产生一个 kernel
    的负载（约 0.1~3 ms），靠循环把它叠成连续负载。

    返回值里额外带两个键：

        reached    True = 已到额定时钟；False = 超时仍未到（**下面的数字不可信**）；
                   None = 读不到时钟（没有 nvidia-smi），退化成固定时长预热
        elapsed_s  实际预热耗时

    为什么盯**显存**时钟而不是 SM 时钟：带宽上限由显存时钟决定。这台机器
    冷态 7001 MHz、稳态 8001 MHz，差 14%，而理论峰值是按 8001 算的。
    SM 时钟（1890 → 2325 MHz）只影响能不能把访存请求及时发出去。

    ## 预热成功不等于"测的时候也是热的"

    预热只保证**开始**测量时机器是热的。测量期间如果因为它自己降温/降频，
    表头就必须反映**测时**的情况 —— 所以 `run_bench.py` 会在每一轮之间再采样一次，
    用 `describe_samples()` 报"测时时钟区间"，而不是只报预热那一刻的值。
    （第一版只报了预热采样，结果出现"表头说没到额定、而表里算出 93% 峰值"这种
    自相矛盾的文件 —— 93% 只有在额定时钟下才可能。）
    """
    t0 = time.time()
    state = query_state()

    if state is None:
        # 读不到时钟：退化成固定时长预热，并如实把 reached 报成 None
        _apply_load(load, fallback_s)
        return {"reached": None, "elapsed_s": time.time() - t0}

    _apply_load(load, min_s)
    state = query_state() or state
    while time.time() - t0 < budget_s and not at_rated_memory_clock(state, tolerance):
        _apply_load(load, burst_s)
        state = query_state() or state

    out = dict(state)
    out["reached"] = at_rated_memory_clock(state, tolerance)
    out["elapsed_s"] = time.time() - t0
    return out


def describe_samples(samples: list[dict]) -> str:
    """把**测量期间**采到的若干次时钟状态汇总成一行。

    比只报预热那一刻更诚实：读者关心的是"产出这些数字的时候，机器是什么状态"。
    """
    mem = [s["mem_clock_mhz"] for s in samples if s.get("mem_clock_mhz") is not None]
    rated = next((s["max_mem_clock_mhz"] for s in samples
                  if s.get("max_mem_clock_mhz") is not None), None)
    if not mem:
        return "测时显存时钟 **未知**（采样失败）"
    lo, hi = min(mem), max(mem)
    if rated is None:
        return f"测时显存时钟 **{lo:.0f}~{hi:.0f} MHz**（额定未知）"
    flag = "满速" if lo >= rated * 0.99 else "**未到额定！**"
    span = f"{lo:.0f}~{hi:.0f}" if lo != hi else f"{hi:.0f}"
    return f"测时显存时钟 **{span} / {rated:.0f} MHz**（{flag}）"


def samples_at_rated(samples: list[dict], tolerance: float = 0.01) -> bool | None:
    """测量期间的采样是否**全程**都在额定时钟上。没有可用采样时返回 None。

    取"最差的那一次"判断：只要有一轮掉到额定以下，这组数字就打折扣。
    """
    flags = [at_rated_memory_clock(s, tolerance) for s in samples]
    flags = [f for f in flags if f is not None]
    if not flags:
        return None
    return all(flags)


def describe(state: dict) -> str:
    """一行人类可读的时钟状态，用来写进基准表头。"""
    mem = state.get("mem_clock_mhz")
    rated = state.get("max_mem_clock_mhz")
    sm = state.get("sm_clock_mhz")
    reached = state.get("reached")
    elapsed = state.get("elapsed_s")

    if mem is None or rated is None:
        mem_text = "显存时钟 **未知**（没有 nvidia-smi，或查询失败）"
    else:
        flag = {True: "满速", False: "**未到额定！**", None: "未知"}[reached]
        mem_text = f"显存时钟 **{mem:.0f} / {rated:.0f} MHz**（{flag}）"

    extras = []
    if sm is not None:
        extras.append(f"SM {sm:.0f} MHz")
    for key, unit in (("temperature_c", "°C"), ("power_w", "W")):
        if state.get(key) is not None:
            extras.append(f"{state[key]:.0f}{unit}")
    if elapsed is not None:
        extras.append(f"预热 {elapsed:.1f}s")

    suffix = f" · {' · '.join(extras)}" if extras else ""
    return mem_text + suffix
