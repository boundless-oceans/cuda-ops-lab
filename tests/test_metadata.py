"""元数据一致性检查（不需要 GPU）。

这是无 GPU 环境下最有价值的一组检查。它能挡住三类**没有任何运行期征兆**的错误：

  * 元数据里把符号名拼错（声明了但扩展里没有）
  * kernel 写了、绑定了，却忘了登记进元数据（导出了但没人用）
  * 加了新变体，却忘了在 VARIANT_NOTES 里写"思路"

这三类错误跑起来都不会崩，只会让某个变体**悄悄不参与测试和基准** ——
等到发现时已经不知道漏了多久。只有对着名单逐条核才能发现。
"""

from __future__ import annotations

import torch

from harness import assert_true, test
from ops_lab import registry as reg


@test("章节模块结构完整", group="meta")
def test_chapter_structure() -> None:
    modules = reg.chapters()
    assert_true(len(modules) > 0, "没有发现任何章节模块")

    problems = []
    for module in modules:
        if not hasattr(module, "VARIANT_NOTES"):
            problems.append(f"{module.NAME} 缺少 VARIANT_NOTES")
        if not getattr(module, "OPS", None):
            problems.append(f"{module.NAME} 的 OPS 是空的")
        # 章节名以编号开头，注册表才能稳定排序
        if not str(module.NAME)[:2].isdigit():
            problems.append(f"{module.NAME} 应以章节编号开头（如 '02_elementwise'）")
    assert_true(not problems, "; ".join(problems))


@test("每个变体都写了 VARIANT_NOTES", group="meta")
def test_every_variant_has_note() -> None:
    missing = []
    for ref in reg.iter_variants():
        if ref.label not in reg.variant_notes(ref.chapter):
            missing.append(str(ref))
    assert_true(not missing, "以下变体没有写思路说明：" + ", ".join(missing))


@test("算子描述符字段齐全", group="meta")
def test_op_spec_fields() -> None:
    problems = []
    for chapter, op, spec in reg.iter_ops():
        for key in ("variants", "reference", "shapes", "bytes", "flops"):
            if key not in spec:
                problems.append(f"{chapter}/{op} 缺字段 {key}")
        if not spec.get("variants"):
            problems.append(f"{chapter}/{op} 没有任何变体")
        if not spec.get("shapes"):
            problems.append(f"{chapter}/{op} 没有测试形状")
    assert_true(not problems, "; ".join(problems))


@test("形状覆盖 0 元素与单元素", group="meta")
def test_shapes_cover_edges() -> None:
    """每个算子都必须有"空输入"和"极小输入"两个测试形状。

    判据是**元素个数**而不是形状元组本身 —— 第 5 章之前所有算子的形状都是一维，
    写 `(0,) in shapes` 就够了；但二维算子的空输入长成 `(0, 5)` 或 `(5, 0)`，
    单元素长成 `(1, 1)`。按元组判会**误报**，按元素个数判对两种都成立。
    """

    def nelem(shape) -> int:
        n = 1
        for d in shape:
            n *= d
        return n

    problems = []
    for chapter, op, spec in reg.iter_ops():
        sizes = {nelem(tuple(s)) for s in spec["shapes"]}
        if 0 not in sizes:
            problems.append(f"{chapter}/{op} 缺 0 元素（空输入）")
        if 1 not in sizes:
            problems.append(f"{chapter}/{op} 缺单元素（极小）")
    assert_true(not problems, "; ".join(problems))


@test("bytes / flops 对每个形状都能算出来", group="meta")
def test_bytes_flops_callable() -> None:
    problems = []
    for chapter, op, spec in reg.iter_ops():
        for shape in spec["shapes"]:
            for key in ("bytes", "flops"):
                try:
                    value = spec[key](shape)
                except Exception as exc:  # noqa: BLE001
                    problems.append(f"{chapter}/{op} {key}{tuple(shape)} 抛异常 {exc!r}")
                    continue
                if not isinstance(value, (int, float)) or value < 0:
                    problems.append(f"{chapter}/{op} {key}{tuple(shape)} = {value!r} 不合法")

            # **逐变体的 bytes 覆盖也要查。**
            #
            # 这一段是被第 4 章逼出来的：那里允许在 variants 声明里覆盖 bytes
            # （三段式 scan 读两遍输入 = 12N、单趟 = 8N）。这类 lambda 写错了
            # （拼错变量、没处理 n=0）只会在**跑基准时**炸，而基准需要 GPU ——
            # 于是错误能潜伏很久。离线检查是唯一能挡住它的地方。
            for label in spec["variants"]:
                try:
                    value = reg.variant_bytes(spec, label, shape)
                except Exception as exc:  # noqa: BLE001
                    problems.append(
                        f"{chapter}/{op}/{label} 的 bytes{tuple(shape)} 抛异常 {exc!r}")
                    continue
                if not isinstance(value, (int, float)) or value < 0:
                    problems.append(
                        f"{chapter}/{op}/{label} 的 bytes{tuple(shape)} = {value!r} 不合法")

        # 变体的实际 bytes 不允许**小于**理想流量 —— 理想是下限，
        # 比它小说明有人把"理想"和"实际"两个方向搞反了
        biggest = max(spec["shapes"], key=lambda s: s[0])
        ideal = spec["bytes"](biggest)
        for label in spec["variants"]:
            actual = reg.variant_bytes(spec, label, biggest)
            if actual < ideal:
                problems.append(
                    f"{chapter}/{op}/{label}: 实际搬运 {actual} < 理想搬运 {ideal}"
                    f"（理想是下限，不可能更少）")
    assert_true(not problems, "; ".join(problems))


@test("参考实现可跑且输出形状正确", group="meta")
def test_reference_runs() -> None:
    """参考实现跑在 **float32 输入升上来的 float64** 上。

    这样比对时误差只来自 kernel 本身，不掺入"输入从 float64 降成 float32"
    的那一份量化误差 —— 否则容差就没法设了。

    输出形状：默认与输入形状相同（elementwise 这类）。**归约这类输出比输入小的
    算子**要在描述符里声明 `out_shape`（如 `lambda shape: (1,)`），否则这条检查
    会把"正确的归约"误判成形状错误。
    """
    problems = []
    for chapter, op, spec in reg.iter_ops():
        n_in = reg.arity(spec)
        out_shape_of = spec.get("out_shape")
        for shape in spec["shapes"]:
            shape = tuple(shape)
            gen = spec.get("gen")
            inputs32 = (
                gen(shape)
                if gen
                else tuple(torch.randn(shape, dtype=torch.float32) for _ in range(n_in))
            )
            inputs64 = tuple(t.double() if torch.is_tensor(t) else t for t in inputs32)
            try:
                out = spec["reference"](*inputs64)
            except Exception as exc:  # noqa: BLE001
                problems.append(
                    f"{chapter}/{op} shape={shape} 参考实现抛异常 "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            expected_shape = tuple(out_shape_of(shape)) if out_shape_of else shape
            if tuple(out.shape) != expected_shape:
                problems.append(
                    f"{chapter}/{op} shape={shape} 参考实现输出形状 {tuple(out.shape)}，"
                    f"期望 {expected_shape}"
                )
    assert_true(not problems, "; ".join(problems))


@test("符号双向一致：元数据 ↔ 扩展导出", group="meta")
def test_symbol_consistency() -> None:
    rep = reg.consistency_report()
    messages = []
    if rep["declared_not_exported"]:
        messages.append(
            "元数据声明了但扩展没导出（符号名拼错？忘了绑定？）："
            + ", ".join(rep["declared_not_exported"])
        )
    if rep["exported_not_declared"]:
        messages.append(
            "扩展导出了但元数据没登记（忘了登记？还是漏写进 UTILITIES？）："
            + ", ".join(rep["exported_not_declared"])
        )
    assert_true(not messages, " | ".join(messages))


@test("每个变体的参数个数与绑定一致", group="meta")
def test_arity_matches_binding() -> None:
    """把"元数据声明了几个输入"和"扩展实际接受几个参数"对起来。

    这条检查是**被真机上的失败逼出来的**：`hello/v2_grid_stride` 声明只传 n，
    但绑定的函数要 (n, block_size, grid_size)，于是真机测试报 TypeError。
    而 `test_generic` 的用例都标了需要 GPU，在没显卡的机器上全被跳过 ——
    这类不一致会一直潜伏，直到有人上真机跑才炸。

    所以必须有一条离线检查把它挡住。pybind11 的签名藏在 __doc__ 首行，
    registry.bound_arity() 负责解析。
    """
    specs = {(chapter, op): spec for chapter, op, spec in reg.iter_ops()}
    problems = []
    for ref in reg.iter_variants():
        spec = specs[(ref.chapter, ref.op)]
        declared_inputs = reg.arity(spec)
        expected = declared_inputs + len(ref.extra_args)
        actual = reg.bound_arity(ref.symbol)
        if actual is None:
            continue  # 解析不出签名就不判，避免误报
        if actual != expected:
            problems.append(
                f"{ref}：元数据声明 {expected} 个参数"
                f"（参考实现 {declared_inputs} 个 + 变体额外 {len(ref.extra_args)} 个），"
                f"但 {ref.symbol} 实际接受 {actual} 个"
            )
    assert_true(not problems, "; ".join(problems))
