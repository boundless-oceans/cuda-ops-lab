"""GPU 时钟状态模块的离线测试（不需要 GPU）。

被这个模块绊过一次，所以它的**解析**部分必须有测试：

第 3 章的两条测量路径对同一个 kernel 给出了 313.3 µs 与 275.5 µs。
根因不是 kernel，而是 `nvidia-smi` 报的显存时钟从冷态 7001 MHz 爬到了
稳态 8001 MHz（差 14%），而理论峰值带宽是按额定 8001 算的 ——
冷机那几次测量的**真实利用率同样是 95.6%**，只是分母用了它当时达不到的时钟。

解析逻辑如果错了（比如把 `[N/A]` 当成 0、或者把额定值当成当前值），
上面那个坑就会被悄悄掩盖过去，而不是被报出来。所以这里逐条钉住：

  * `[N/A]` / 空字段 → None（**不是 0**，0 会被误读成"时钟掉到 0"）
  * 小数、正整数、带单位后缀都要能解析
  * 缺字段的行不抛异常
  * `at_rated_memory_clock` 在拿不到数据时返回 None，而不是猜 True
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent
for _p in (str(_REPO_ROOT / "python"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ops_lab import clock_state  # noqa: E402

from harness import assert_true, test  # noqa: E402

# 真机实测的两行（2026-09，RTX 4060 Laptop）
COLD = "1890 MHz, 7001 MHz, 2100 MHz, 8001 MHz, 41, 9.85 W"
WARM = "2325 MHz, 8001 MHz, 2100 MHz, 8001 MHz, 47, 55.21 W"


@test("解析 nvidia-smi 的一行采样", group="meta")
def test_parse_state() -> None:
    cold = clock_state.parse_state(COLD)
    assert_true(cold["sm_clock_mhz"] == 1890.0, f"SM 时钟解析错：{cold['sm_clock_mhz']}")
    assert_true(cold["mem_clock_mhz"] == 7001.0, f"显存时钟解析错：{cold['mem_clock_mhz']}")
    assert_true(cold["max_mem_clock_mhz"] == 8001.0,
                f"额定显存时钟解析错：{cold['max_mem_clock_mhz']}")
    assert_true(cold["temperature_c"] == 41.0, f"温度解析错：{cold['temperature_c']}")
    assert_true(abs(cold["power_w"] - 9.85) < 1e-9, f"功耗解析错：{cold['power_w']}")

    warm = clock_state.parse_state(WARM)
    assert_true(warm["power_w"] == 55.21, f"功耗小数解析错：{warm['power_w']}")


@test("N/A 与空字段解析成 None 而不是 0", group="meta")
def test_parse_missing_fields() -> None:
    """`[N/A]` 必须变成 None —— 变成 0 会被读成"时钟掉到 0"，是另一种谎。"""
    state = clock_state.parse_state("1890 MHz, 7001 MHz, [N/A], [N/A], [N/A], [N/A]")
    assert_true(state["mem_clock_mhz"] == 7001.0, "能解析的字段不该受影响")
    for key in ("max_sm_clock_mhz", "max_mem_clock_mhz", "temperature_c", "power_w"):
        assert_true(state[key] is None, f"{key} 应解析成 None，实际 {state[key]!r}")

    # 字段比预期少也不能抛异常（nvidia-smi 版本差异）
    short = clock_state.parse_state("1890 MHz")
    assert_true(short["sm_clock_mhz"] == 1890.0, "只有第一个字段时也该解析出来")
    assert_true(short["mem_clock_mhz"] is None, "缺的字段应为 None")


@test("时钟是否到额定值的判断", group="meta")
def test_at_rated_memory_clock() -> None:
    assert_true(clock_state.at_rated_memory_clock(clock_state.parse_state(WARM)) is True,
                "8001/8001 应判为已到额定")
    assert_true(clock_state.at_rated_memory_clock(clock_state.parse_state(COLD)) is False,
                "7001/8001 应判为未到额定（这正是第 3 章那 14% 的来源）")
    # 1% 容差：7990/8001 也算到了
    near = clock_state.parse_state("2325 MHz, 7990 MHz, 2100 MHz, 8001 MHz, 47, 55 W")
    assert_true(clock_state.at_rated_memory_clock(near) is True, "1% 以内算已到额定")
    # 拿不到数据时必须返回 None —— 猜 True 会让"未预热"悄悄通过
    assert_true(clock_state.at_rated_memory_clock({}) is None, "没有数据应返回 None")
    assert_true(
        clock_state.at_rated_memory_clock({"mem_clock_mhz": 7001.0}) is None,
        "只有当前值、没有额定值时也应返回 None",
    )


@test("describe 会把未到额定的状态如实说出来", group="meta")
def test_describe() -> None:
    text = clock_state.describe(dict(clock_state.parse_state(WARM), reached=True,
                                    elapsed_s=1.5))
    assert_true("8001" in text and "满速" in text, f"满速时应写明显存时钟与满速：{text}")

    text = clock_state.describe(dict(clock_state.parse_state(COLD), reached=False,
                                    elapsed_s=9.0))
    assert_true("未到额定" in text, f"未到额定时必须警告（这是那个坑的信号）：{text}")

    # 没有 nvidia-smi 时也不能崩，要把"未知"说出来
    text = clock_state.describe({"reached": None, "elapsed_s": 2.0})
    assert_true("未知" in text, f"拿不到时钟时应写明未知：{text}")


@test("describe_samples 汇总的是测量期间的时钟区间", group="meta")
def test_describe_samples() -> None:
    warm = [clock_state.parse_state(WARM)] * 3
    text = clock_state.describe_samples(warm)
    assert_true("8001" in text and "满速" in text, f"全程额定时应写满速：{text}")

    # 只要有一轮掉下来，就要如实报出区间并警告 —— 这正是第一版漏掉的情况
    mixed = [clock_state.parse_state(WARM), clock_state.parse_state(COLD)]
    text = clock_state.describe_samples(mixed)
    assert_true("7001~8001" in text, f"应报出 7001~8001 的区间：{text}")
    assert_true("未到额定" in text, f"区间跨到额定以下时必须警告：{text}")

    assert_true("未知" in clock_state.describe_samples([]), "没有采样时应写未知")
    assert_true("未知" in clock_state.describe_samples([{}]),
                "采样里没有时钟字段时也应写未知")


@test("samples_at_rated 取最差的一次，拿不到就返回 None", group="meta")
def test_samples_at_rated() -> None:
    warm = clock_state.parse_state(WARM)
    cold = clock_state.parse_state(COLD)
    assert_true(clock_state.samples_at_rated([warm, warm]) is True, "全程额定应判 True")
    assert_true(clock_state.samples_at_rated([warm, cold]) is False,
                "只要有一轮没到额定就必须判 False —— 这组数字要打折扣")
    assert_true(clock_state.samples_at_rated([]) is None, "没有采样应返回 None")
    assert_true(clock_state.samples_at_rated([{}, {}]) is None,
                "采样里没有时钟字段时应返回 None，而不是猜 True")


@test("预热必须连续施压，而不是跑一次查一次", group="meta")
def test_apply_load_duty_cycle() -> None:
    """回归测试 —— 第一版预热就是这么写错的。

    第一版是 `while ...: load(); query_state(); time.sleep(0.2)`。
    `query_state()` 要走一次 nvidia-smi 子进程（几十毫秒），加上 sleep，
    GPU 的**占空比掉到千分之一** —— 实测"预热 8 秒，时钟一动不动"
    （表头采样到 SM 1890 MHz / 9 W，一台空转的机器），于是产出了
    "表头说没到额定、表里却算出 93% 峰值"这种自相矛盾的结果文件。

    修法是把负载**连续**压一段再查。这里钉住"连续"这条性质：
    0.2 秒内必须压很多次，而不是被查询/sleep 稀释成个位数。
    """
    import time

    stamps: list[float] = []
    t0 = time.perf_counter()
    clock_state._apply_load(lambda: stamps.append(time.perf_counter()), 0.2)
    elapsed = time.perf_counter() - t0

    assert_true(elapsed >= 0.2, f"应当至少压满 0.2 秒，实际 {elapsed:.3f}")
    assert_true(len(stamps) > 50,
                f"0.2 秒内只压了 {len(stamps)} 次：负载被查询/sleep 稀释了，"
                f"这正是第一版占空比 bug 的症状")
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    if gaps:
        assert_true(max(gaps) < 0.05,
                    f"两次负载之间最长隔了 {max(gaps) * 1e3:.0f} ms —— 中间夹了查询或睡眠")
