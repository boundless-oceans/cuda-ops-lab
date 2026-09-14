#!/usr/bin/env python3
"""基准：跑出优化阶梯表。

用法：

    python bench/run_bench.py                          # 全部章节
    python bench/run_bench.py --chapters elementwise   # 只跑第 2 章
    python bench/run_bench.py --warmup 20 --iters 100
    python bench/run_bench.py --peak-gbps 256          # 自动探测失败时手动指定
    python bench/run_bench.py --out /tmp/bench.md      # 全部输出到一个文件

输出：打印到屏幕，同时每个章节写一份 `bench/results/<章节名>.md`（该目录已 gitignore）。

退出码：0 = 正常；2 = 看不到 GPU（基准测的就是真实耗时，没有替代方案）。

## 计时方法（这里错了，后面所有数字都是垃圾）

    1. **先 warmup**：第一次启动 kernel 要建立上下文、加载模块，那段耗时不是
       稳态性能。
    2. **每次迭代单独 record/sync**：这样拿到的是独立样本，一次偶发抖动只污染
       一个样本，取 median 就能把它剔掉。代价是每次多 ~5µs 同步开销 ——
       基准形状的耗时在几百微秒量级，可以忽略。
    3. **取 median 而不是平均**：平均值会被离群值拖走。
    4. **预分配**：输入在计时循环外就搬到 GPU 上；循环里只启动 kernel。

异步陷阱（`docs/operator_code_navigation.md` §2.1 讲过）：`kernel<<<>>>` 返回时
GPU 还没开始跑。不 record/sync 就计时，测到的是"启动 kernel 花了多久"。
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))

import torch  # noqa: E402

from ops_lab import device_query, registry  # noqa: E402
from ops_lab.envguard import sanitize_syspath  # noqa: E402

BENCH_SEED = 20260914
DEFAULT_WARMUP = 20
DEFAULT_ITERS = 100
DEFAULT_RESULT_DIR = _REPO_ROOT / "bench" / "results"


# ============================================================ 纯函数（可离线测试）


def gbps_from(bytes_count: int, ms: float) -> float:
    """搬运 bytes_count 字节耗时 ms 毫秒 → 有效带宽 GB/s。"""
    if ms <= 0:
        return 0.0
    return bytes_count / (ms * 1e-3) / 1e9


def percent_of_peak(gbps: float, peak_gbps: float | None) -> float | None:
    """占理论峰值的百分比；峰值未知时返回 None（而不是除以零或瞎猜）。"""
    if not peak_gbps or peak_gbps <= 0:
        return None
    return 100.0 * gbps / peak_gbps


def theoretical_ms(bytes_count: int, peak_gbps: float | None) -> float | None:
    """理论最短耗时（毫秒）。这是每张阶梯表的判据来源。"""
    if not peak_gbps or peak_gbps <= 0:
        return None
    return bytes_count / (peak_gbps * 1e9) * 1e3


# ==================================================================== 计时


@dataclass
class Measurement:
    label: str
    note: str
    median_ms: float
    gbps: float
    percent_peak: float | None
    iters: int
    is_baseline: bool = False


def _make_inputs(spec: dict, shape: tuple, device: torch.device) -> tuple:
    """造输入。固定种子，保证每次跑的数字可比。"""
    torch.manual_seed(BENCH_SEED)
    n_in = registry.arity(spec)
    gen = spec.get("gen")
    if gen is not None:
        inputs = gen(shape)
    else:
        inputs = tuple(torch.randn(shape, dtype=torch.float32) for _ in range(n_in))
    return tuple(t.to(device) if torch.is_tensor(t) else t for t in inputs)


def _time(fn, warmup: int, iters: int) -> float:
    """warmup + iters 次独立计时，返回 median（毫秒）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def measure_variant(ref: registry.VariantRef, spec: dict, shape: tuple,
                    warmup: int, iters: int, peak: float | None) -> Measurement:
    device = torch.device("cuda")
    inputs = _make_inputs(spec, shape, device)
    median = _time(lambda: registry.call(ref, *inputs), warmup, iters)
    gbps = gbps_from(spec["bytes"](shape), median)
    return Measurement(
        label=ref.label,
        note=registry.variant_notes(ref.chapter).get(ref.label, ""),
        median_ms=median,
        gbps=gbps,
        percent_peak=percent_of_peak(gbps, peak),
        iters=iters,
    )


def measure_baseline(spec: dict, shape: tuple, warmup: int,
                     iters: int, peak: float | None) -> Measurement | None:
    fn = spec.get("torch")
    if fn is None:
        return None
    device = torch.device("cuda")
    inputs = _make_inputs(spec, shape, device)
    median = _time(lambda: fn(*inputs), warmup, iters)
    gbps = gbps_from(spec["bytes"](shape), median)
    return Measurement(
        label="torch（基线）",
        note="torch 的对应实现",
        median_ms=median,
        gbps=gbps,
        percent_peak=percent_of_peak(gbps, peak),
        iters=iters,
        is_baseline=True,
    )


# ==================================================================== 报告


def render_op(spec: dict, op: str, shape: tuple, measurements: list[Measurement],
              baseline: Measurement | None, peak: float | None) -> str:
    nbytes = spec["bytes"](shape)
    flops = spec["flops"](shape)
    intensity = (flops / nbytes) if nbytes else 0.0
    t_ms = theoretical_ms(nbytes, peak)

    lines = [f"### `{op}`  shape=`{tuple(shape)}`", ""]
    # 不写死"（读 + 写）"：hello 这类算子只写不读，写死就是错的。
    # 读写比例由各章 README 说明；metadata 里的 bytes 是总搬运量。
    lines.append(f"- 搬运 **{nbytes / 1e6:.1f} MB**，运算 {flops:,} FLOP，"
                 f"算术强度 **{intensity:.4f} FLOP/Byte**")
    if t_ms is not None:
        lines.append(
            f"- 理论最短耗时 = {nbytes / 1e6:.1f} MB ÷ {peak:.1f} GB/s = **{t_ms * 1e3:.1f} µs**"
        )
    else:
        lines.append("- 理论最短耗时：**未知**（拿不到峰值带宽，用 `--peak-gbps` 手动指定）")
    lines.append("")
    lines.append("| 变体 | 思路 | median (µs) | GB/s | %峰值 | 相比上一级 |")
    lines.append("|---|---|---|---|---|---|")

    previous_ms: float | None = None
    for m in measurements:
        speedup = f"{previous_ms / m.median_ms:.2f}×" if previous_ms else "—"
        pct = f"{m.percent_peak:.1f}%" if m.percent_peak is not None else "?"
        lines.append(
            f"| `{m.label}` | {m.note} | {m.median_ms * 1e3:.1f} | "
            f"{m.gbps:.1f} | {pct} | {speedup} |"
        )
        previous_ms = m.median_ms

    if baseline is not None:
        pct = f"{baseline.percent_peak:.1f}%" if baseline.percent_peak is not None else "?"
        fastest = min(measurements, key=lambda x: x.median_ms)
        ratio = fastest.median_ms / baseline.median_ms
        lines.append(
            f"| **{baseline.label}** | {baseline.note} | {baseline.median_ms * 1e3:.1f} | "
            f"{baseline.gbps:.1f} | {pct} | 最快变体是它的 {ratio:.2f}× |"
        )

    lines.append("")
    if t_ms is not None:
        fastest = min(measurements, key=lambda x: x.median_ms)
        lines.append(
            f"**最快变体 `{fastest.label}`**：{fastest.median_ms * 1e3:.1f} µs，"
            f"是理论下限的 **{fastest.median_ms / t_ms:.2f}×**"
            f"（即还有 {max(0.0, (1 - t_ms / fastest.median_ms)) * 100:.0f}% 的空间）。"
        )
        lines.append("")
    return "\n".join(lines)


def render_chapter(chapter: str, blocks: list[str], peak: float | None,
                   device_name: str | None) -> str:
    peak_text = f"{peak:.1f} GB/s" if peak else "**未知**"
    lines = [f"# {chapter} 优化阶梯表", ""]
    lines.append(f"- 设备：{device_name or '未知'}")
    lines.append(f"- 理论峰值带宽：{peak_text}")
    lines.append("- 计时：warmup 后逐次 record/sync，取 median")
    lines.append("")
    lines.extend(blocks)
    return "\n".join(lines)


# ==================================================================== 入口


def main() -> int:
    ap = argparse.ArgumentParser(description="cuda-ops-lab 基准")
    ap.add_argument("--chapters", nargs="*", default=None,
                    help="只跑章节名包含这些词的（如 execution elementwise）")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--peak-gbps", type=float, default=None,
                    help="手动指定理论峰值带宽；不给则用 nvidia-smi 探测")
    ap.add_argument("--out", type=Path, default=None,
                    help="把所有输出写到这一个文件；默认每章写 bench/results/<章节名>.md")
    ap.add_argument("--quiet", action="store_true", help="不打印到屏幕，只写文件")
    args = ap.parse_args()

    sanitize_syspath()

    if not torch.cuda.is_available():
        print("看不到 CUDA 设备，无法跑基准。", file=sys.stderr)
        print("基准测的就是真实耗时，没有 GPU 就没有替代方案。", file=sys.stderr)
        print("（离线能做的检查请用：python tests/run_all.py）", file=sys.stderr)
        return 2

    peak = args.peak_gbps
    if peak is None:
        peak = device_query.theoretical_bandwidth_gbps()
        if peak is None:
            print("警告：拿不到峰值带宽（nvidia-smi 解析失败），%峰值一列会显示 ?。",
                  file=sys.stderr)
            print("      可以用 --peak-gbps 手动指定。", file=sys.stderr)

    name = device_query.device_name() or torch.cuda.get_device_name(0)

    # 按章节归拢
    per_chapter: dict[str, list[str]] = {}
    for chapter, op, spec in registry.iter_ops():
        if args.chapters and not any(w in chapter for w in args.chapters):
            continue

        bench_shapes = spec.get("bench_shapes") or [max(spec["shapes"], key=lambda s: s[0])]
        for shape in bench_shapes:
            shape = tuple(shape)
            print(f"[bench] {chapter}/{op} shape={shape} ...", file=sys.stderr)

            measurements = [
                measure_variant(ref, spec, shape, args.warmup, args.iters, peak)
                for ref in registry.iter_variants()
                if ref.chapter == chapter and ref.op == op
            ]
            baseline = measure_baseline(spec, shape, args.warmup, args.iters, peak)
            per_chapter.setdefault(chapter, []).append(
                render_op(spec, op, shape, measurements, baseline, peak)
            )

    if not per_chapter:
        print("没有匹配的章节", file=sys.stderr)
        return 1

    for chapter, blocks in per_chapter.items():
        text = render_chapter(chapter, blocks, peak, name)
        if not args.quiet:
            print()
            print(text)
        out = args.out or (DEFAULT_RESULT_DIR / f"{chapter}.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"[bench] 已写入 {out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
