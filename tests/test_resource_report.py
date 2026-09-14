"""静态资源报告脚本的解析测试（不需要 GPU）。

`scripts/resource_report.py` 整体标着"不需要 GPU"，但它的**解析器**尤其值得
单独测：ptxas 的输出格式是外部契约，解析错了会静默地给出错误的寄存器数 ——
而"错误的数字"比"没有数字"危险得多，因为它看起来是正常的。

下面的样本是从本机真实 ptxas 输出里抄的（含长 mangled name 与 `cmem[0]` 后缀）。
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for _p in (str(_SCRIPTS), str(_SCRIPTS.parent / "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from harness import assert_true, test  # noqa: E402

import resource_report as rr  # noqa: E402

# 真实样本：两个 kernel，其中一个名字很长
_SAMPLE = """ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_ZN7ops_lab11elementwise62_GLOBAL__N__7fd756c8_29_elementwise_v0_uncoalesced_cu_edcf410213add_v0_kernelEPKfS3_Pfl' for 'sm_89'
ptxas info    : Function properties for _ZN7ops_lab11elementwise62_GLOBAL__N__7fd756c8_29_elementwise_v0_uncoalesced_cu_edcf410213add_v0_kernelEPKfS3_Pfl
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, 384 bytes cmem[0]
ptxas info    : Compiling entry function '_ZN7ops_lab11elementwise62_GLOBAL__N__7fd756c8_29_elementwise_v0_uncoalesced_cu_edcf410215unary_v0_kernelINS0_9SigmoidOpEEEvPKfPfl' for 'sm_89'
ptxas info    : Function properties for _ZN7ops_lab11elementwise62_GLOBAL__N__7fd756c8_29_elementwise_v0_uncoalesced_cu_edcf410215unary_v0_kernelINS0_9SigmoidOpEEEvPKfPfl
    32 bytes stack frame, 16 bytes spill stores, 8 bytes spill loads
ptxas info    : Used 40 registers, 1024 bytes smem, 400 bytes cmem[0]
"""


@test("parse_ptxas 解析出全部 kernel", group="meta")
def test_parse_count() -> None:
    records = rr.parse_ptxas(_SAMPLE)
    assert_true(len(records) == 2, f"应解析出 2 条记录，实际 {len(records)}")

    first = records[0]
    assert_true(first[0].endswith("add_v0_kernelEPKfS3_Pfl"), "第一条应是 add_v0_kernel")
    assert_true(first[1] == "sm_89", f"架构应为 sm_89，实际 {first[1]}")
    assert_true(first[2] == 12, f"寄存器应为 12，实际 {first[2]}")
    assert_true(first[3] == 0, f"stack 应为 0，实际 {first[3]}")
    assert_true(first[6] == 0, f"无 smem 字段时应为 0，实际 {first[6]}")


@test("parse_ptxas 正确处理 spill 与 smem", group="meta")
def test_parse_spill_and_smem() -> None:
    second = rr.parse_ptxas(_SAMPLE)[1]
    assert_true(second[2] == 40, f"寄存器应为 40，实际 {second[2]}")
    assert_true(second[3] == 32, f"stack 应为 32，实际 {second[3]}")
    assert_true(second[4] == 16, f"spill stores 应为 16，实际 {second[4]}")
    assert_true(second[5] == 8, f"spill loads 应为 8，实际 {second[5]}")
    # 这一条验证 "1024 bytes smem" 能被从 "Used 40 registers, ..." 里挑出来
    assert_true(second[6] == 1024, f"smem 应为 1024，实际 {second[6]}")


@test("parse_ptxas 对残缺输入不谎报 0", group="meta")
def test_parse_missing_stats() -> None:
    """统计行缺失时必须返回 -1（未知），不能返回 0。

    0 看起来像"没问题"，而未知是"需要人来查" —— 两者混同会让报告骗人。
    """
    partial = "ptxas info    : Compiling entry function '_Z3foov' for 'sm_89'\n"
    records = rr.parse_ptxas(partial)
    assert_true(len(records) == 1, "应解析出 1 条记录")
    assert_true(records[0][2] == -1, f"寄存器未知时应为 -1，实际 {records[0][2]}")
    assert_true(records[0][3] == -1, f"stack 未知时应为 -1，实际 {records[0][3]}")


@test("demangle 能从含匿名命名空间的名字里取出裸函数名", group="meta")
def test_demangle_strips_anonymous_namespace() -> None:
    """这是实测踩过的坑：反混淆结果里的 `(anonymous namespace)` **自带括号**，
    用 `split("(")[0]` 截取会在那里被切断，最后回退成 mangled 名。"""
    mangled = (
        "_ZN7ops_lab11elementwise62_GLOBAL__N__7fd756c8_29_elementwise_v0_uncoalesced"
        "_cu_edcf410213add_v0_kernelEPKfS3_Pfl"
    )
    assert_true(
        mangled[:11] == "_ZN7ops_lab", "样本本身应当是 C++ mangled 名"
    )
    names = rr.demangle([mangled])
    got = names.get(mangled, "")
    assert_true(got == "add_v0_kernel", f"应取出 add_v0_kernel，实际 {got!r}")


@test("demangle 对空输入与非法输入都不炸", group="meta")
def test_demangle_robustness() -> None:
    assert_true(rr.demangle([]) == {}, "空输入应返回空字典")
    # 非 mangled 的普通名字应当原样返回，而不是消失
    out = rr.demangle(["not_a_mangled_name"])
    assert_true(out.get("not_a_mangled_name") == "not_a_mangled_name",
                f"普通名字应原样返回，实际 {out}")


@test("demangle 不会切进模板实参", group="meta")
def test_demangle_keeps_template_args() -> None:
    """第二个坑：模板实参里含 `::`。

    `unary_v0_kernel<ops_lab::elementwise::SigmoidOp>` 按 `::` 取最后一段
    会得到 `SigmoidOp>` —— 名字被切断了。所以必须按尖括号深度切。
    """
    # 直接测 bare_name，不依赖 c++filt 的输出格式
    cases = [
        (
            "ops_lab::elementwise::(anonymous namespace)::add_v0_kernel(float const*, float*, long)",
            "add_v0_kernel",
        ),
        (
            "ops_lab::elementwise::(anonymous namespace)::unary_v0_kernel<ops_lab::elementwise::SigmoidOp>(float const*, float*, long)",
            "unary_v0_kernel<SigmoidOp>",
        ),
        (
            "ops_lab::execution::(anonymous namespace)::copy_smem_kernel(float4 const*, float4*, float const*, float*, long, long)",
            "copy_smem_kernel",
        ),
        ("plain_function(int)", "plain_function"),
    ]
    for demangled, expected in cases:
        got = rr.bare_name(demangled)
        assert_true(got == expected, f"输入 {demangled[:50]}... 应得到 {expected!r}，实际 {got!r}")
