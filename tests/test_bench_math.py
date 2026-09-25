"""基准脚本的换算公式测试（不需要 GPU）。

`bench/run_bench.py` 的计时部分只能在真机上跑，但**单位换算**是纯函数 ——
而换算错了整张阶梯表的数字就全错，所以必须单独测。

这类"离线可测的一部分"很容易被忽略：整个脚本标着"需要 GPU"，于是它的
每一行都被跳过了。但换算公式不需要显卡。
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent
for _p in (str(_REPO_ROOT / "bench"), str(_REPO_ROOT / "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from harness import assert_true, test  # noqa: E402

import run_bench  # noqa: E402


@test("gbps_from 换算正确", group="meta")
def test_gbps_from() -> None:
    # 256 GB/s 的机器上搬 256 MB 应当刚好 1 ms
    assert_true(
        abs(run_bench.gbps_from(256_000_000, 1.0) - 256.0) < 1e-6,
        f"256MB / 1ms 应为 256 GB/s，实际 {run_bench.gbps_from(256_000_000, 1.0)}",
    )
    # 教科书例子：201.3 MB / 786 µs ≈ 256 GB/s
    got = run_bench.gbps_from(201_326_592, 0.786)
    assert_true(abs(got - 256.1) < 0.5, f"期望约 256 GB/s，实际 {got:.1f}")
    # 除以零必须返回 0 而不是抛异常（计时失败时不该把整个基准带崩）
    assert_true(run_bench.gbps_from(1000, 0.0) == 0.0, "ms=0 时应返回 0")


@test("percent_of_peak 处理峰值未知的情况", group="meta")
def test_percent_of_peak() -> None:
    assert_true(run_bench.percent_of_peak(128.0, 256.0) == 50.0, "128/256 应为 50%")
    assert_true(run_bench.percent_of_peak(256.0, 256.0) == 100.0, "满速应为 100%")
    # 峰值拿不到时必须是 None（表格里显示 ?），而不是 0 或除零崩溃
    assert_true(run_bench.percent_of_peak(128.0, None) is None, "峰值未知应返回 None")
    assert_true(run_bench.percent_of_peak(128.0, 0.0) is None, "峰值为 0 应返回 None")


@test("theoretical_ms 与 gbps_from 互为逆运算", group="meta")
def test_theoretical_ms_roundtrip() -> None:
    nbytes, peak = 201_326_592, 256.0
    ms = run_bench.theoretical_ms(nbytes, peak)
    assert_true(ms is not None, "峰值已知时不应为 None")
    assert_true(abs(ms - 0.7864) < 1e-3, f"201.3MB / 256GB/s 应约 0.786 ms，实际 {ms:.4f}")
    # 拿理论耗时反推带宽，应当回到峰值
    back = run_bench.gbps_from(nbytes, ms)
    assert_true(abs(back - peak) < 1e-6, f"往返换算应当回到 {peak}，实际 {back}")
    assert_true(run_bench.theoretical_ms(nbytes, None) is None, "峰值未知应返回 None")


@test("「实际搬运」列只在各变体字节数不同时才出现", group="meta")
def test_render_op_bytes_column() -> None:
    """第 4 章 scan 逼出来的新列：三段式读两遍输入（12N）、单趟只读一遍（8N）。

    出现条件必须是"各变体实际搬运量不同" —— 否则前几章每张表都会多出一个常数
    列，那是纯噪声。这条规则决定了所有章节的表长什么样，所以值得钉住。
    """
    spec = {"bytes": lambda shape: 8 * shape[0], "flops": lambda shape: shape[0]}
    shape = (1024,)

    def m(label: str, nbytes: int):
        return run_bench.Measurement(label=label, note="", median_ms=0.1, gbps=100.0,
                                     percent_peak=50.0, iters=40, nbytes=nbytes)

    # 各变体不同 → 有这一列，并且给出与理想搬运的倍数
    differing = run_bench.render_op(spec, "op", shape, [m("a", 8192), m("b", 12288)],
                                    None, 256.0)
    assert_true("| 实际搬运 |" in differing, f"字节数不同时应当有这一列：\n{differing}")
    assert_true("1.50×" in differing, f"应当给出相对理想搬运的倍数：\n{differing}")

    # 各变体相同 → 没有这一列（比如第 2、3 章）
    same = run_bench.render_op(spec, "op", shape, [m("a", 8192), m("b", 8192)], None, 256.0)
    assert_true("| 实际搬运 |" not in same, f"字节数相同时不该多出常数列：\n{same}")


@test("阶梯表的每一行列数都与表头一致", group="meta")
def test_render_op_column_count() -> None:
    """**被一个真实 bug 逼出来的测试**：加"实际搬运"列时我用字符串片段拼行，
    少写了一个 `|`，于是产出"多一个空单元格、且 GB/s 与搬运量挤在同一格"的表
    —— 数字一个没丢，所以肉眼扫过去很像对的，但**表结构已经坏了**。

    这类错误是纯文本层面的，离线就能查：表头有几个 `|` 分隔的列，
    每一行就必须有同样多个。
    """
    spec = {"bytes": lambda shape: 8 * shape[0], "flops": lambda shape: shape[0]}
    shape = (1024,)

    def m(label: str, nbytes: int):
        return run_bench.Measurement(label=label, note="", median_ms=0.1, gbps=100.0,
                                     percent_peak=50.0, iters=40, nbytes=nbytes)

    def check(text: str, what: str) -> None:
        rows = [ln for ln in text.splitlines() if ln.startswith("|")]
        assert_true(len(rows) >= 3, f"{what}: 表格行数太少：\n{text}")
        header_cols = rows[0].count("|") - 1  # | a | b | → 首尾各一个竖线
        for ln in rows[1:]:
            if set(ln) <= set("|- "):  # 分隔行
                continue
            cols = ln.count("|") - 1
            assert_true(cols == header_cols,
                        f"{what}: 表头 {header_cols} 列，这一行 {cols} 列：\n{ln}")
            assert_true("| |" not in ln, f"{what}: 出现空单元格（分隔符拼错了）：\n{ln}")

    # 两条路径都要查：字节数相同（第 1~3 章的形态）与不同（第 4 章）
    check(run_bench.render_op(spec, "op", shape, [m("a", 8192), m("b", 12288)], None, 256.0),
          "字节数不同")
    check(run_bench.render_op(spec, "op", shape, [m("a", 8192), m("b", 8192)], None, 256.0),
          "字节数相同")
