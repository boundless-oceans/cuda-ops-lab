"""cuda-ops-lab 的 Python 包。

当前（M1/S0）只包含基础设施：

* ``envguard``      —— 剔除外来 sys.path 条目
* ``build_config``  —— 构建参数探测（check_env 与 _build 的唯一事实来源）

M1/S1 会在这里补上惰性扩展加载（``_extension``）与算子元数据注册表。
"""

from . import build_config, envguard

__all__ = ["build_config", "envguard"]
