#!/usr/bin/env python3
"""内核静态资源报告：寄存器 / spill / 共享内存。

用法：

    python scripts/resource_report.py                     # 全部 kernel
    python scripts/resource_report.py --chapter elementwise
    python scripts/resource_report.py --out build/resource_report.md

**不需要 GPU**，也不需要先跑构建 —— 它自己用 nvcc 编译一遍，只为了拿 ptxas 的
统计输出。

## 为什么要用它，而不是看构建日志

`_build.py -v` 也能看到同样的数字，但有两个问题：ninja 是增量的，没变化的文件
不会重新编译，因此**没有输出**；而且输出混在几百行编译命令里，没法一眼看出
"哪个变体寄存器最多、有没有 spill"。

这份报告把全部 kernel 拉平成一张表，和第 5 步的带宽阶梯表并排看，就能回答
"这一版慢，是不是因为寄存器爆了/在 spill"。

## 编译参数来自 build_config

**这一点是刻意的**：报告必须用和正式构建**完全相同**的参数编译，
否则报告里的寄存器数和真实构建出来的产物可能对不上，报告就成了装饰品。
所以这里不硬编码任何 flag，全部取自 `ops_lab.build_config.detect()`。

## 一个已知的盲区

**动态共享内存不会出现在这份报告里。** ptxas 的静态统计看不到
`extern __shared__` 的用量 —— 那是**启动参数**，不是编译期属性。
第 1 章 `bandwidth_probe` 的四个变体靠动态 smem 区分 occupancy，
在这里会显示 smem 全为 0，这是正常的（验证方法见
`docs/operator_code_navigation.md` §5：看 SASS 里有没有 STS/LDS）。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))

from ops_lab import build_config  # noqa: E402
from ops_lab.envguard import sanitize_syspath  # noqa: E402

REPORT_DIR = _REPO_ROOT / "build" / "resource_report"


# ==================================================================== 解析


@dataclass
class KernelResources:
    source: str          # 相对 kernels/ 的路径
    kernel: str          # 反混淆后的裸函数名
    registers: int
    stack_bytes: int
    spill_bytes: int     # spill stores + spill loads
    smem_bytes: int      # 静态共享内存（动态 smem 不在此列，见模块注释）
    cmem_bytes: int
    arch: str


_COMPILE_RE = re.compile(r"^'([^']+)'\s+for\s+'([^']+)'")
_PROPS_RE = re.compile(
    r"Function properties for \S+\s*\n"
    r"\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads"
)
_USAGE_RE = re.compile(r"Used (\d+) registers,(.*)")


def parse_ptxas(text: str) -> list[tuple[str, str, int, int, int, int, int]]:
    """解析 `nvcc -Xptxas -v` 的 stderr。

    返回 [(mangled_name, arch, registers, stack, spill_stores, spill_loads, smem), ...]

    按 "Compiling entry function" 切块，每块里找对应的统计行。
    刻意**不做宽松匹配**：解析不出来就少一条记录，由调用方报告出来 ——
    静默地记成 0 比缺一条更危险，因为 0 看起来像"没问题"。
    """
    out = []
    blocks = re.split(r"ptxas info\s*:\s*Compiling entry function", text)
    for block in blocks[1:]:
        head = _COMPILE_RE.match(block.strip())
        if not head:
            continue
        mangled, arch = head.group(1), head.group(2)

        props = _PROPS_RE.search(block)
        # 统计行缺失时用 -1 标记"未知"，而不是 0
        stack = int(props.group(1)) if props else -1
        spill_stores = int(props.group(2)) if props else -1
        spill_loads = int(props.group(3)) if props else -1

        usage = _USAGE_RE.search(block)
        registers = int(usage.group(1)) if usage else -1
        smem = 0
        if usage:
            m = re.search(r"(\d+) bytes smem", usage.group(2))
            smem = int(m.group(1)) if m else 0

        out.append((mangled, arch, registers, stack, spill_stores, spill_loads, smem))
    return out


def bare_name(demangled: str) -> str:
    """从反混淆结果里取出裸函数名：保留模板实参，去掉命名空间与参数列表。

    看着简单，实际有两个坑（都是实测踩出来的）：

    1. **`(anonymous namespace)` 自带括号** —— 直接 `split("(")[0]` 会在这里
       被切断，得到 `ops_lab::elementwise:`，再取 `::` 最后一段就是空串，
       最后回退成 mangled 名，整张表全是乱码。
    2. **模板实参里含 `::`** —— 例如
       `unary_v0_kernel<ops_lab::elementwise::SigmoidOp>(...)`，
       按 `::` 取最后一段会切进模板参数内部，得到 `SigmoidOp>`。

    所以两处切分都必须按**尖括号深度**来做，不能靠朴素字符串切分。
    """
    text = demangled.replace("(anonymous namespace)::", "").replace("{anonymous}::", "")

    # 第一刀：参数列表的起始括号（深度 0 的那个）
    depth = 0
    cut = len(text)
    for i, ch in enumerate(text):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            cut = i
            break
    head = text[:cut]

    # 第二刀：从右往左找命名空间分隔符，只看深度 0 的 "::"
    depth = 0
    result = head
    for i in range(len(head) - 1, -1, -1):
        ch = head[i]
        if ch == ">":
            depth += 1
        elif ch == "<":
            depth -= 1
        elif ch == ":" and depth == 0 and i > 0 and head[i - 1] == ":":
            result = head[i + 1 :]
            break

    # 第三刀（纯可读性）：外层命名空间已经剥掉了，剩下的 `::` 前缀只可能出现在
    # 模板实参里。把它们也去掉，`unary_v0_kernel<ops_lab::elementwise::SigmoidOp>`
    # 变成 `unary_v0_kernel<SigmoidOp>`。
    return re.sub(r"(?:[A-Za-z_]\w*::)+", "", result).strip()


def demangle(names: list[str]) -> dict[str, str]:
    """用 c++filt 批量反混淆，并只保留裸函数名。

    `ops_lab::elementwise::(anonymous namespace)::add_v1_naive(float const*, ...)`
    → `add_v1_naive`

    这样表里一眼能看出是哪个 kernel，不用去读 90 个字符的 mangled name。
    """
    if not names:
        return {}
    try:
        proc = subprocess.run(
            ["c++filt"], input="\n".join(names), capture_output=True, text=True, timeout=30
        )
        demangled = proc.stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return {n: n for n in names}

    if len(demangled) != len(names):
        return {n: n for n in names}

    out = {}
    for mangled, pretty in zip(names, demangled):
        out[mangled] = bare_name(pretty) or mangled
    return out


# ==================================================================== 编译


def collect_sources(chapter: str | None) -> list[Path]:
    sources = sorted((_REPO_ROOT / "kernels").rglob("*.cu"))
    if chapter:
        sources = [s for s in sources if chapter in str(s.relative_to(_REPO_ROOT))]
    return sources


def compile_and_parse(cfg: build_config.BuildConfig, source: Path,
                      verbose: bool) -> tuple[list, str]:
    """编译单个 .cu 并解析 ptxas 输出。返回 (记录列表, 错误信息)。"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    obj = REPORT_DIR / (source.stem + ".o")

    cmd = [str(cfg.nvcc)] + cfg.nvcc_compile_flags() + ["-c", str(source), "-o", str(obj)]
    if verbose:
        print(f"[resource_report] {' '.join(cmd)}", file=sys.stderr)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"nvcc 调用失败：{exc}"

    if proc.returncode != 0:
        # 只留前几行错误，避免刷屏
        head = "\n".join(proc.stderr.strip().splitlines()[:5])
        return [], f"编译失败（nvcc 退出码 {proc.returncode}）：\n{head}"

    return parse_ptxas(proc.stderr), ""


# ==================================================================== 报告


def render(records: list[KernelResources], arch: str, failures: list[tuple[str, str]]) -> str:
    lines = ["# 内核静态资源报告", ""]
    lines.append(f"- 目标架构：`{arch}`")
    lines.append("- 编译参数与正式构建**完全相同**（同一份 `build_config`）")
    lines.append("- **不需要 GPU**")
    lines.append("")
    lines.append(f"共 {len(records)} 个 kernel。")
    lines.append("")
    lines.append("| 源文件 | kernel | 寄存器 | stack (B) | spill (B) | smem (B) |")
    lines.append("|---|---|---|---|---|---|")

    for r in records:
        reg = str(r.registers) if r.registers >= 0 else "?"
        spill = str(r.spill_bytes) if r.spill_bytes >= 0 else "?"
        stack = str(r.stack_bytes) if r.stack_bytes >= 0 else "?"
        lines.append(
            f"| `{r.source}` | `{r.kernel}` | {reg} | {stack} | {spill} | {r.smem_bytes} |"
        )
    lines.append("")

    # --- 值得注意的 ---
    notes = []
    spilled = [r for r in records if r.spill_bytes > 0]
    if spilled:
        notes.append("**有 spill 的 kernel**（变量被塞进 local memory，每次访问要走显存）：")
        for r in spilled:
            notes.append(f"- `{r.source}` / `{r.kernel}`：spill {r.spill_bytes} B")
    else:
        notes.append("**没有任何 kernel 出现 spill。**")

    if records:
        worst = max(records, key=lambda r: r.registers)
        notes.append(
            f"\n寄存器最多的是 `{worst.kernel}`（{worst.registers} 个）—— 寄存器越多，"
            f"occupancy 上限越低，两者要一起看。"
        )

    unknown = [r for r in records if r.registers < 0 or r.spill_bytes < 0]
    if unknown:
        notes.append(
            f"\n⚠️ 有 {len(unknown)} 个 kernel 的统计行没解析出来（表里显示 ?）。"
            f"这**不等于**没问题，是解析器需要更新。"
        )

    lines.append("## 值得注意的")
    lines.append("")
    lines.extend(notes)
    lines.append("")

    if failures:
        lines.append("## 编译失败")
        lines.append("")
        for src, err in failures:
            lines.append(f"### `{src}`")
            lines.append("")
            lines.append("```")
            lines.append(err)
            lines.append("```")
            lines.append("")

    lines.append("## 一个已知盲区")
    lines.append("")
    lines.append(
        "**动态共享内存不出现在这张表里。** ptxas 的静态统计看不到 `extern __shared__` "
        "的用量 —— 那是启动参数，不是编译期属性。第 1 章 `bandwidth_probe` 的四个变体"
        "靠动态 smem 区分 occupancy，在这里 smem 全为 0 是正常的。"
    )
    lines.append("")
    return "\n".join(lines)


# ==================================================================== 入口


def main() -> int:
    ap = argparse.ArgumentParser(description="内核静态资源报告（寄存器 / spill / smem）")
    ap.add_argument("--chapter", default=None, help="只报告路径里含这个词的源文件")
    ap.add_argument("--out", type=Path, default=None, help="写到文件；默认打印到屏幕")
    ap.add_argument("-v", "--verbose", action="store_true", help="打印每个编译命令")
    args = ap.parse_args()

    sanitize_syspath(verbose=args.verbose)

    cfg = build_config.detect()
    if cfg.has_fatal():
        print("环境有致命问题，无法编译：", file=sys.stderr)
        for problem in cfg.problems:
            print(f"  X {problem}", file=sys.stderr)
        print("\n先运行：python scripts/check_env.py", file=sys.stderr)
        return 1

    sources = collect_sources(args.chapter)
    if not sources:
        print("没有匹配的 .cu 文件", file=sys.stderr)
        return 1

    records: list[KernelResources] = []
    failures: list[tuple[str, str]] = []

    for source in sources:
        rel = str(source.relative_to(_REPO_ROOT))
        print(f"[resource_report] {rel}", file=sys.stderr)

        raw, error = compile_and_parse(cfg, source, args.verbose)
        if error:
            failures.append((rel, error))
            continue

        mangled_names = [r[0] for r in raw]
        pretty = demangle(mangled_names)
        for mangled, arch, regs, stack, sp_st, sp_ld, smem in raw:
            records.append(
                KernelResources(
                    source=rel,
                    kernel=pretty.get(mangled, mangled),
                    registers=regs,
                    stack_bytes=stack,
                    spill_bytes=(sp_st + sp_ld) if (sp_st >= 0 and sp_ld >= 0) else -1,
                    smem_bytes=smem,
                    cmem_bytes=0,
                    arch=arch,
                )
            )

    text = render(records, cfg.arch, failures)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"[resource_report] 已写入 {args.out}", file=sys.stderr)
    else:
        print(text)

    # 有失败就以非 0 退出，便于接进 CI
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
