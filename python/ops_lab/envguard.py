"""把"外来"的 sys.path 条目剔除掉，保证入口脚本在任何 shell 下行为一致。

## 背景（实测，不是假设）

本机全局 `PYTHONPATH` 是：

    /opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages

也就是 ROS2 Humble 的包目录，装在**系统 python3.10** 上，与本仓库无关。
但 `PYTHONPATH` 里的路径会被无条件塞进**任何** Python 解释器的 `sys.path`，
包括我们用的 python3.9。后果是：

* 纯 Python 包（`yaml`、`typing_extensions` 之类）可能被意外从那里导入；
* 带 `.so` 的包会报 "invalid ELF header" / "module compiled for 3.10" 这类错误；
* 最麻烦的是**同一个脚本在不同终端里行为不同** —— 因为 `PYTHONPATH` 取决于
  你有没有 `source /opt/ros/humble/setup.bash`。

这类问题排查成本极高（错误信息指向的位置看起来完全无辜），所以在入口脚本
最早的位置直接消除掉。

## 判断依据

不硬编码 "/opt/ros"，而是两条通用规则：

1. **版本不匹配**：路径里带 `pythonX.Y` 且与当前解释器版本不同 → 外来。
   这条规则换机器也不会误伤。
2. **已知外部工作空间前缀**：目前只有 `/opt/ros/`。

其余一律保留，避免误删用户自己的 `PYTHONPATH` 配置。
"""

from __future__ import annotations

import os
import re
import sys

__all__ = ["sanitize_syspath", "foreign_reason"]

_PYVER_RE = re.compile(r"python(\d)\.(\d+)")
_FOREIGN_PREFIXES = ("/opt/ros/",)


def foreign_reason(path: str) -> str | None:
    """判断 path 是否应该从 sys.path 剔除。

    返回剔除原因（人类可读），不需要剔除则返回 None。
    """
    if not path:
        return None

    for prefix in _FOREIGN_PREFIXES:
        if path.startswith(prefix):
            return f"属于已知外部工作空间（前缀 {prefix}）"

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
    removed: list[tuple[str, str]] = []
    kept: list[str] = []

    for p in sys.path:
        reason = foreign_reason(p)
        if reason is None:
            kept.append(p)
        else:
            removed.append((p, reason))

    if not removed:
        return removed

    sys.path[:] = kept

    # 同步 PYTHONPATH：ninja / 子进程 继承的是环境变量而不是 sys.path，
    # 不同步的话"父进程干净、子进程脏"，问题会更隐蔽。
    entries = [e for e in os.environ.get("PYTHONPATH", "").split(os.pathsep) if e]
    kept_env = [e for e in entries if foreign_reason(e) is None]
    if kept_env:
        os.environ["PYTHONPATH"] = os.pathsep.join(kept_env)
    else:
        os.environ.pop("PYTHONPATH", None)

    if verbose:
        print(f"[envguard] 已从 sys.path 剔除 {len(removed)} 个外来路径：", file=sys.stderr)
        for p, reason in removed:
            print(f"[envguard]   - {p}\n[envguard]     原因：{reason}", file=sys.stderr)
    else:
        first = removed[0][0]
        more = f"（等共 {len(removed)} 个）" if len(removed) > 1 else ""
        # 走 stderr：诊断信息不能污染 stdout，否则 `--json` 这类机器可读输出会被破坏
        print(f"[envguard] 已剔除外来 sys.path 路径：{first} {more}", file=sys.stderr)
        print("[envguard] 加 -v 查看详情；这是 PYTHONPATH 污染，不是本仓库的 bug", file=sys.stderr)

    return removed
