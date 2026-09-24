#!/usr/bin/env python3
"""基准：跑出优化阶梯表。

用法：

    python bench/run_bench.py                          # 全部章节
    python bench/run_bench.py --chapters elementwise   # 只跑第 2 章
    python bench/run_bench.py --rounds 5 --iters 30    # 多测几轮 / 每轮少测几次
    python bench/run_bench.py --peak-gbps 256          # 自动探测失败时手动指定
    python bench/run_bench.py --out /tmp/bench.md      # 全部输出到一个文件

输出：打印到屏幕，同时每个章节写一份 `bench/results/<章节名>.md`（该目录已 gitignore）。

退出码：0 = 正常；2 = 看不到 GPU（基准测的就是真实耗时，没有替代方案）。

## 计时方法（这里错了，后面所有数字都是垃圾）

    0. **先预热到时钟稳态**（★ 2026-09 补上，见 `ops_lab/clock_state.py`）。
       这台机器的**显存时钟**冷态 7001 MHz、稳态 8001 MHz，差 **14%**；而理论峰值
       带宽是按额定 8001 MHz 算的。冷机开测会让最先测的几个变体被打上
       "只有 83% 峰值"的标签 —— 它其实是 95.6%，只是分母用了当时达不到的时钟。
       **这一步不做，整张表都不可信。**
    1. **每次测量前 warmup**：建立上下文、暖缓存，让第一批样本不是冷启动。
    2. **每次迭代单独 record/sync**：拿到独立样本，一次偶发抖动只污染一个样本。
       代价是每次多 ~5 µs 同步开销 —— 基准形状的耗时在几百微秒量级，可以忽略。
    3. **取 median 而不是平均**：平均值会被离群值拖走。
    4. **交错测量**（★ 2026-09 补上）：一轮里把所有变体各测一遍，轮间把起始变体
       错开，最后取"各轮 median 的中位数"。残留的慢漂移（温度、降频）会均匀落在
       所有变体上，而不是全压在排在前面那几个身上。表里的 **轮间极差** 就是给读者
       看的稳定性指标：它超过百分之几，这一格就别当结论用。
    5. **同一份输入给所有变体**：不再"每个变体各造一份数据"，少一个变量。
    6. **预分配**：输入在计时循环外就搬到 GPU 上；循环里只启动 kernel。

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

from ops_lab import clock_state, device_query, registry  # noqa: E402
from ops_lab.envguard import sanitize_syspath  # noqa: E402

BENCH_SEED = 20260914
DEFAULT_WARMUP = 20
DEFAULT_ITERS = 40
DEFAULT_ROUNDS = 3
DEFAULT_RESULT_DIR = _REPO_ROOT / "bench" / "results"


# ============================================================ 退出码是个坑
#
# 本仓库约定 **退出码 2 = "跳过"**（`scripts/run_all.sh` 靠它区分"没 GPU"和"失败"）。
# 而 argparse 遇到**用法错误**（参数拼错、缺值）时默认也是 `exit(2)` ——
# 于是 `run_bench.py --chaptres reduction` 这种手滑会被 run_all.sh 汇报成
# "SKIP"，看起来和"这台机器没显卡"一模一样：**基准根本没跑，却没有任何失败**。
#
# 实测确认过：拼错一个字母 → 退出码 2。所以这里把用法错误的退出码改成 1，
# 让"跳过"在结构上只可能来自"真的没有 GPU"。
class _StrictParser(argparse.ArgumentParser):
    def error(self, message: str):  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: 参数错误：{message}\n")


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
    rounds: int = 1
    spread_ms: float = 0.0  # 各轮 median 的极差 —— 给读者看的稳定性指标


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


def _rotate(items: list, offset: int) -> list:
    """把列表左移 offset 位。用来让每一轮从不同的变体开始测。"""
    if not items:
        return items
    k = offset % len(items)
    return items[k:] + items[:k]


def measure_op(spec: dict, refs: list[registry.VariantRef], shape: tuple, rounds: int,
               warmup: int, iters: int, peak: float | None,
               clock_samples: list[dict] | None = None) -> tuple[list[Measurement],
                                                                 Measurement | None]:
    """交错测量一个算子的全部变体 + torch 对照。

    返回 `(变体表, 基线)`。每个变体的 median 是"各轮 median 的中位数"，
    极差是各轮 median 的最大差 —— 后者是这个数字可不可信的唯一线索。

    `clock_samples` 非空时，每轮结束采一次时钟并追加进去。**这是为了留下证据**：
    表头要报的是"产出这些数字时机器处于什么状态"，而不是"预热那一刻的状态"。
    （第一版只报预热采样，结果产出了"表头说没到额定、表里却算出 93% 峰值"
    这种自相矛盾的文件 —— 93% 只有在额定时钟下才可能出现。）
    """
    device = torch.device("cuda")
    inputs = _make_inputs(spec, shape, device)  # 一份数据，所有变体、所有轮共用
    nbytes = spec["bytes"](shape)

    per_round: dict[str, list[float]] = {ref.label: [] for ref in refs}
    baseline_rounds: list[float] = []
    baseline_fn = spec.get("torch")

    for r in range(rounds):
        for ref in _rotate(list(refs), r):
            per_round[ref.label].append(_time(lambda: registry.call(ref, *inputs), warmup, iters))
        if baseline_fn is not None:
            baseline_rounds.append(_time(lambda: baseline_fn(*inputs), warmup, iters))
        if clock_samples is not None:
            state = clock_state.query_state()
            if state is not None:
                clock_samples.append(state)

    notes = registry.variant_notes(refs[0].chapter) if refs else {}

    def make(label: str, note: str, values: list[float], baseline: bool = False) -> Measurement:
        med = statistics.median(values)
        gbps = gbps_from(nbytes, med)
        return Measurement(
            label=label,
            note=note,
            median_ms=med,
            gbps=gbps,
            percent_peak=percent_of_peak(gbps, peak),
            iters=iters,
            is_baseline=baseline,
            rounds=rounds,
            spread_ms=max(values) - min(values),
        )

    measurements = [make(ref.label, notes.get(ref.label, ""), per_round[ref.label])
                    for ref in refs]
    baseline = None
    if baseline_rounds:
        baseline = make("torch（基线）", "torch 的对应实现", baseline_rounds, baseline=True)
    return measurements, baseline


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
    lines.append("| 变体 | 思路 | median (µs) | 轮间极差 (µs) | GB/s | %峰值 | 相比上一级 |")
    lines.append("|---|---|---|---|---|---|---|")

    previous_ms: float | None = None
    for m in measurements:
        speedup = f"{previous_ms / m.median_ms:.2f}×" if previous_ms else "—"
        pct = f"{m.percent_peak:.1f}%" if m.percent_peak is not None else "?"
        lines.append(
            f"| `{m.label}` | {m.note} | {m.median_ms * 1e3:.1f} | {m.spread_ms * 1e3:.1f} | "
            f"{m.gbps:.1f} | {pct} | {speedup} |"
        )
        previous_ms = m.median_ms

    if baseline is not None:
        pct = f"{baseline.percent_peak:.1f}%" if baseline.percent_peak is not None else "?"
        fastest = min(measurements, key=lambda x: x.median_ms)
        ratio = fastest.median_ms / baseline.median_ms
        lines.append(
            f"| **{baseline.label}** | {baseline.note} | {baseline.median_ms * 1e3:.1f} | "
            f"{baseline.spread_ms * 1e3:.1f} | {baseline.gbps:.1f} | {pct} | "
            f"最快变体是它的 {ratio:.2f}× |"
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
                   device_name: str | None, clock_lines: list[str], clock_ok: bool | None,
                   rounds: int, warmup: int, iters: int) -> str:
    peak_text = f"{peak:.1f} GB/s" if peak else "**未知**"
    lines = [f"# {chapter} 优化阶梯表", ""]
    lines.append(f"- 设备：{device_name or '未知'}")
    lines.append(f"- 理论峰值带宽：{peak_text}")
    # 时钟必须写进表头：它是"这一列数字可不可信"的唯一线索。
    # 分成"预热"和"测时"两行 —— 前者说明准备得怎么样，后者才是**产出这些数字时**
    # 机器的状态。只报前者会产出自相矛盾的文件（见 measure_op 的注释）。
    lines.extend(f"- {line}" for line in clock_lines)
    lines.append(f"- 计时：{rounds} 轮交错（轮间起始变体错开），每轮 warmup {warmup} + "
                 f"{iters} 次独立 record/sync；取各轮 median 的中位数")
    if clock_ok is False:
        lines.append("")
        lines.append("> ⚠️ **测时显存时钟没有到额定值** —— %峰值 那一列的分母是按额定时钟算的，"
                     "因此会**系统性偏低**。这组数字不要直接采信："
                     "先把机器跑热（或关掉其它占用 GPU 的进程）再重测。")
    lines.append("")
    lines.extend(blocks)
    return "\n".join(lines)


# ==================================================================== 入口


def _steady_state_load(chapters: list[str] | None):
    """构造一个"能压住 GPU"的可调用对象：反复跑第一个匹配到的变体。

    用算子本身来预热，而不是另写一个专门的加热 kernel —— 预热负载的访存特征
    与真实测量一致，而且不需要多维护一份代码。
    """
    for chapter, op, spec in registry.iter_ops():
        if chapters and not any(w in chapter for w in chapters):
            continue
        for ref in registry.iter_variants():
            if ref.chapter == chapter and ref.op == op:
                shapes = spec.get("bench_shapes") or [max(spec["shapes"], key=lambda s: s[0])]
                inputs = _make_inputs(spec, tuple(shapes[0]), torch.device("cuda"))
                return lambda: registry.call(ref, *inputs)
    return None


def main() -> int:
    # allow_abbrev=False：不接受 `--chapter` 这种"前缀碰巧唯一"的缩写。
    # 缩写会随选项增删而改变行为（今天能跑、明天变成歧义错误），而基准脚本的
    # 调用方是文档和 shell 脚本 —— 那里的命令必须是确定的。
    ap = _StrictParser(description="cuda-ops-lab 基准", allow_abbrev=False)
    ap.add_argument("--chapters", nargs="*", default=None,
                    help="只跑章节名包含这些词的（如 execution elementwise）")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                    help="每轮每次测量前的预热次数（默认 20）")
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS,
                    help="每轮每个变体的计时次数（默认 40）；总样本数 = 轮数 × 本值")
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                    help="交错测量的轮数（默认 3）：轮间起始变体错开，抵消慢漂移")
    ap.add_argument("--peak-gbps", type=float, default=None,
                    help="手动指定理论峰值带宽；不给则用 nvidia-smi 探测")
    ap.add_argument("--out", type=Path, default=None,
                    help="把所有输出写到这一个文件；默认每章写 bench/results/<章节名>.md")
    ap.add_argument("--quiet", action="store_true", help="不打印到屏幕，只写文件")
    ap.add_argument("--no-warmup", action="store_true",
                    help="跳过时钟稳态预热（**只用于复现那个坑**，正常测速别加）")
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

    # ---- 时钟稳态预热：不做这一步，最先测的变体会被凭空打上"83%" ----
    clock_samples: list[dict] = []
    if args.no_warmup:
        state = clock_state.query_state() or {}
        reached = clock_state.at_rated_memory_clock(state) if state else None
        warmup_line = "预热：**已跳过**（--no-warmup）· " + clock_state.describe(
            dict(state, reached=reached))
        print(f"[bench] 跳过了稳态预热：{warmup_line}", file=sys.stderr)
    else:
        load = _steady_state_load(args.chapters)
        if load is None:
            print("没有匹配的章节", file=sys.stderr)
            return 1
        print("[bench] 预热到时钟稳态 ……", file=sys.stderr)
        state = clock_state.steady_state(load)
        warmup_line = "预热：" + clock_state.describe(state)
        print(f"[bench] {warmup_line}", file=sys.stderr)
        if state.get("reached") is False:
            print("警告：预热结束后显存时钟仍未到额定值；测量期间还会再采样确认。",
                  file=sys.stderr)

    # 按章节归拢
    per_chapter: dict[str, list[str]] = {}
    for chapter, op, spec in registry.iter_ops():
        if args.chapters and not any(w in chapter for w in args.chapters):
            continue

        refs = [ref for ref in registry.iter_variants()
                if ref.chapter == chapter and ref.op == op]
        bench_shapes = spec.get("bench_shapes") or [max(spec["shapes"], key=lambda s: s[0])]
        for shape in bench_shapes:
            shape = tuple(shape)
            print(f"[bench] {chapter}/{op} shape={shape} ...", file=sys.stderr)

            measurements, baseline = measure_op(
                spec, refs, shape, args.rounds, args.warmup, args.iters, peak, clock_samples)
            per_chapter.setdefault(chapter, []).append(
                render_op(spec, op, shape, measurements, baseline, peak)
            )

    if not per_chapter:
        print("没有匹配的章节", file=sys.stderr)
        return 1

    # 表头报**测时**的时钟，而不是预热那一刻的：读者关心的是"产出这些数字时
    # 机器处于什么状态"。只要有一轮采样掉到额定以下，就按"不可信"警告。
    clock_ok = clock_state.samples_at_rated(clock_samples)
    if clock_ok is None:
        clock_ok = state.get("reached") if not args.no_warmup else reached
    clock_lines = [warmup_line, clock_state.describe_samples(clock_samples)]
    print(f"[bench] {clock_lines[1]}", file=sys.stderr)

    for chapter, blocks in per_chapter.items():
        text = render_chapter(chapter, blocks, peak, name, clock_lines, clock_ok,
                              args.rounds, args.warmup, args.iters)
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
