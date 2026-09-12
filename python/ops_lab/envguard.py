"""把"外来"的 sys.path 条目剔除掉，保证入口脚本在任何 shell 下行为一致。

## 背景

某些 shell 环境会通过 ``PYTHONPATH`` 注入**其它工作空间**的包目录（典型来源是
机器人中间件、工具链的 ``setup`` 脚本），而这些目录常常属于**另一个 Python 版本**。
``PYTHONPATH`` 里的路径会被无条件塞进任何 Python 解释器的 ``sys.path``，
包括我们正在用的这个。后果是：

* 纯 Python 包可能被意外从那里导入；
* 带 ``.so`` 的包会报 "invalid ELF header"、"module compiled for X.Y" 之类的错误；
* 最麻烦的是**同一个脚本在不同终端里行为不同** —— 因为 ``PYTHONPATH`` 取决于
  你有没有 source 过那个 setup 脚本。

这类问题排查成本极高（报错指向的位置看起来完全无辜），所以在入口脚本最早的
位置直接消除掉。

## 判断规则

1. **Python 版本不匹配**（默认规则，与机器无关）
   路径里带 ``pythonX.Y`` 且与当前解释器版本不同 → 判定为外来。
   这一条不依赖任何机器特定的路径，换机器也不会误伤。

2. **已知外部工作空间前缀**（可选，默认关闭）
   通过环境变量 ``OPS_LAB_FOREIGN_PATH_PREFIXES`` 提供，多个前缀用
   ``os.pathsep``（Linux 上是冒号）分隔。

   典型用法：当你的机器上有某个工具链的包目录**恰好与你当前的 Python 版本相同**、
   因而规则 1 抓不到它时，用它显式屏蔽::

       export OPS_LAB_FOREIGN_PATH_PREFIXES=/opt/some_toolchain/

   **仓库代码里不硬编码任何机器特定的路径**，需要屏蔽什么由使用者自己声明。

其余路径一律保留，避免误删用户有意配置的 ``PYTHONPATH``。
"""

from __future__ import annotations

import os
import re
import sys

__all__ = ["sanitize_syspath", "foreign_reason"]

_PYVER_RE = re.compile(r"python(\d)\.(\d+)")

# 环境变量名：额外需要屏蔽的外部工作空间前缀（冒号分隔）
PREFIX_ENV_VAR = "OPS_LAB_FOREIGN_PATH_PREFIXES"


def _extra_prefixes() -> tuple[str, ...]:
    """读取使用者通过环境变量声明的额外前缀。默认返回空元组。"""
    raw = os.environ.get(PREFIX_ENV_VAR, "")
    return tuple(p for p in raw.split(os.pathsep) if p)


def foreign_reason(path: str, extra_prefixes: tuple[str, ...] = ()) -> str | None:
    """判断 path 是否应该从 sys.path 剔除。

    参数
        path:            待判断的路径
        extra_prefixes:  额外屏蔽的前缀（由调用方传入，见 ``_extra_prefixes``）

    返回剔除原因（人类可读），不需要剔除则返回 None。
    """
    if not path:
        return None

    for prefix in extra_prefixes:
        if path.startswith(prefix):
            return f"命中使用者声明的外部工作空间前缀 {prefix}"

    m = _PYVER_RE.search(path)
    if m:
        dir_ver = (int(m.group(1)), int(m.group(2)))
        cur_ver = (sys.version_info.major, sys.version_info.minor)
        if dir_ver != cur_ver:
            return (f"Python 版本不匹配（该目录是 {dir_ver[0]}.{dir_ver[1]}，"
                    f"当前解释器是 {cur_ver[0]}.{cur_ver[1]}）")

    return None


def sanitize_syspath(verbose: bool = False) -> list[tuple[str, str]]:
    """剔除外来 sys.path 条目，并同步清理 PYTHONPATH（让子进程也一致）。

    参数
        verbose: True 时逐条打印被剔除的路径和原因；False 时只在有剔除发生时
                 打印一行摘要（因为"发生过剔除"本身是重要信息，静默会让人困惑）。

    返回
        [(被剔除的路径, 原因), ...]，没有剔除时返回空列表。
    """
    prefixes = _extra_prefixes()
    removed: list[tuple[str, str]] = []
    kept: list[str] = []

    for p in sys.path:
        reason = foreign_reason(p, prefixes)
        if reason is None:
            kept.append(p)
        else:
            removed.append((p, reason))

    if not removed:
        return removed

    sys.path[:] = kept

    # 同步 PYTHONPATH：ninja / 其它子进程继承的是环境变量而不是 sys.path，
    # 不同步的话"父进程干净、子进程脏"，问题会更隐蔽。
    entries = [e for e in os.environ.get("PYTHONPATH", "").split(os.pathsep) if e]
    kept_env = [e for e in entries if foreign_reason(e, prefixes) is None]
    if kept_env:
        os.environ["PYTHONPATH"] = os.pathsep.join(kept_env)
    else:
        os.environ.pop("PYTHONPATH", None)

    # 走 stderr：诊断信息不能污染 stdout，否则 `--json` 这类机器可读输出会被破坏
    if verbose:
        print(f"[envguard] 已从 sys.path 剔除 {len(removed)} 个外来路径：", file=sys.stderr)
        for p, reason in removed:
            print(f"[envguard]   - {p}\n[envguard]     原因：{reason}", file=sys.stderr)
    else:
        first = removed[0][0]
        more = f"（等共 {len(removed)} 个）" if len(removed) > 1 else ""
        print(f"[envguard] 已剔除外来 sys.path 路径：{first} {more}", file=sys.stderr)
        print(
            "[envguard] 这通常来自 PYTHONPATH 注入的其它工作空间；"
            "加 -v 看详情，或把该前缀写进 OPS_LAB_FOREIGN_PATH_PREFIXES",
            file=sys.stderr,
        )

    return removed
