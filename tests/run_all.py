#!/usr/bin/env python3
"""跑所有测试。

用法：

    python tests/run_all.py                          # 全部
    python tests/run_all.py --chapters elementwise   # 只跑第 2 章（meta 组始终运行）
    python tests/run_all.py -v                       # 列出每个通过的用例
    python tests/run_all.py --list                   # 只列用例，不执行

退出码：0 = 没有失败（跳过不算失败）；1 = 有失败。

测试模块靠 `@test` 装饰器**自注册**，本文件只负责把它们导进来再跑一遍 ——
所以新增一个 `tests/test_*.py` 不需要改这里。
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

# --- 引导：与其它入口脚本一致，先定位仓库、再净化 sys.path、最后导入 ---
_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent

sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_TESTS_DIR))  # 让 `import harness` 在直接运行时也能工作

from ops_lab.envguard import sanitize_syspath  # noqa: E402

import harness  # noqa: E402


def load_test_modules() -> list[str]:
    loaded = []
    for path in sorted(_TESTS_DIR.glob("test_*.py")):
        importlib.import_module(path.stem)
        loaded.append(path.stem)
    return loaded


def main() -> int:
    ap = argparse.ArgumentParser(description="cuda-ops-lab 测试入口")
    ap.add_argument(
        "--chapters",
        nargs="*",
        default=None,
        help="只跑分组名包含这些词的用例（如 execution elementwise）；meta 组始终运行",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="列出每个通过的用例")
    ap.add_argument("--list", action="store_true", help="只列出用例，不执行")
    args = ap.parse_args()

    sanitize_syspath(verbose=args.verbose)

    modules = load_test_modules()
    all_cases = harness.cases()

    print("=" * 68)
    print(" cuda-ops-lab 测试")
    print("=" * 68)
    print(f"  测试模块  {', '.join(modules)}")
    print(f"  用例总数  {len(all_cases)}")
    print(f"  GPU 可见  {harness.has_gpu()}")
    if args.chapters:
        print(f"  章节过滤  {' '.join(args.chapters)}")

    if args.list:
        print()
        for case in all_cases:
            tag = "  [需要 GPU]" if case.gpu else ""
            print(f"    {case.group:16s} {case.name}{tag}")
        return 0

    print()
    results = harness.run(chapters=args.chapters, verbose=args.verbose)
    failures = harness.report(results, verbose=args.verbose)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
