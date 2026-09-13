"""第 1 章特有的测试（需要 GPU）。

通用正确性由 `test_generic.py` 覆盖（hello 的输出、bandwidth_probe 的拷贝）。
这里只补这一章**特有**的两件事：occupancy 实验的前提是否成立，以及自动 grid 是否可用。
"""

from __future__ import annotations

import torch

from harness import assert_true, test
from ops_lab import _extension

GROUP = "01_execution"


@test("occupancy 随动态共享内存增加而下降", group=GROUP, gpu=True)
def test_occupancy_decreases_with_smem() -> None:
    """这是第 1 章实验的**前提**。如果它不成立，整个实验就没意义。

    设计上四个变体是同一份拷贝逻辑，只有动态共享内存不同（0 / 24 / 32 / 48 KB）。
    占用率必须单调不增，否则说明共享内存根本没被分配 ——
    而那正是"编译器把 extern __shared__ 优化掉"的失效模式。
    """
    ext = _extension.get()
    occ = [ext.execution_probe_occupancy(v, 256)["occupancy"] for v in range(1, 5)]
    smem = [ext.execution_probe_occupancy(v, 256)["dynamic_smem_bytes"] for v in range(1, 5)]

    assert_true(
        smem == sorted(smem),
        f"共享内存声明值不是递增的：{smem}",
    )
    assert_true(
        all(occ[i] >= occ[i + 1] for i in range(len(occ) - 1)),
        f"occupancy 未随共享内存增加而单调不增：{occ}（smem={smem}）",
    )
    assert_true(
        occ[0] > occ[-1],
        f"48KB 变体与 0KB 变体的占用率相同（{occ}），共享内存可能没有真正生效",
    )


@test("hello v2 的 grid_size=0 能自动选择", group=GROUP, gpu=True)
def test_hello_auto_grid() -> None:
    """grid_size=0 表示"按设备规模自动选"。这条路径依赖 cudaGetDeviceProperties，
    离线测不到，只能在真机上确认它不报错、且结果正确。"""
    ext = _extension.get()
    out = ext.execution_hello_v2_grid_stride(4096, 256, 0)
    expected = torch.arange(4096, dtype=torch.int32, device="cuda")
    assert_true(bool(torch.equal(out, expected)), "自动 grid 下 hello v2 的输出不正确")


@test("hello v2 接受显式 grid_size", group=GROUP, gpu=True)
def test_hello_explicit_grid() -> None:
    ext = _extension.get()
    for grid in (1, 3, 64):
        out = ext.execution_hello_v2_grid_stride(1000, 128, grid)
        expected = torch.arange(1000, dtype=torch.int32, device="cuda")
        assert_true(
            bool(torch.equal(out, expected)),
            f"grid_size={grid} 时 hello v2 的输出不正确（循环步进应该覆盖全部元素）",
        )
