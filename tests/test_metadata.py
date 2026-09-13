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


@test("形状覆盖 n=0 与 n=1", group="meta")
def test_shapes_cover_edges() -> None:
    problems = []
    for chapter, op, spec in reg.iter_ops():
        shapes = {tuple(s) for s in spec["shapes"]}
        if (0,) not in shapes:
            problems.append(f"{chapter}/{op} 缺 n=0（空输入）")
        if (1,) not in shapes:
            problems.append(f"{chapter}/{op} 缺 n=1（极小）")
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
    assert_true(not problems, "; ".join(problems))


@test("参考实现可跑且输出形状正确", group="meta")
def test_reference_runs() -> None:
    """参考实现跑在 **float32 输入升上来的 float64** 上。

    这样比对时误差只来自 kernel 本身，不掺入"输入从 float64 降成 float32"
    的那一份量化误差 —— 否则容差就没法设了。
    """
    problems = []
    for chapter, op, spec in reg.iter_ops():
        n_in = reg.arity(spec)
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
            if tuple(out.shape) != shape:
                problems.append(
                    f"{chapter}/{op} shape={shape} 参考实现输出形状 {tuple(out.shape)}，期望 {shape}"
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
