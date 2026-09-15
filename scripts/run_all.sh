#!/usr/bin/env bash
#
# 一键跑完整条链路：体检 → 构建 → 测试 → 静态资源 → native → 基准。
#
# 用法：
#     bash scripts/run_all.sh                    # 全部（基准需要 GPU）
#     bash scripts/run_all.sh --quick            # 跳过需要 GPU 的步骤
#     bash scripts/run_all.sh --chapter elementwise
#     bash scripts/run_all.sh --bench-only
#
# 设计说明：
#   * **不失败即停**，而是跑完所有能跑的步骤再汇总 —— 一次运行拿到全部信息，
#     比修一个错跑一次快得多。
#   * 「没有 GPU」不算失败。基准脚本返回 2 表示"跳过"，脚本会单独归类。
#   * 用的是当前 PATH 上的 python，不写死 conda 环境名（那是各人自己的选择）。
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CHAPTER=""
QUICK=0
BENCH_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)      QUICK=1; shift ;;
    --bench-only) BENCH_ONLY=1; shift ;;
    --chapter)    CHAPTER="${2:-}"; shift 2 ;;
    -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
    *)            echo "未知参数：$1（-h 看用法）" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------- 小工具

declare -a STEP_NAME=() STEP_RESULT=() STEP_SECONDS=()
FAILED=0

hr() { printf '%s\n' "------------------------------------------------------------"; }

run_step() {
  local name="$1"; shift
  hr
  printf '>>> %s\n' "$name"
  hr
  local start=$SECONDS
  "$@"
  local code=$?
  local elapsed=$((SECONDS - start))

  STEP_NAME+=("$name")
  STEP_SECONDS+=("$elapsed")
  if [[ $code -eq 0 ]]; then
    STEP_RESULT+=("OK")
  elif [[ $code -eq 2 ]]; then
    # 约定：2 = 跳过（例如没有 GPU）
    STEP_RESULT+=("SKIP")
  else
    STEP_RESULT+=("FAIL($code)")
    FAILED=1
  fi
  printf '\n（%s：%ds）\n\n' "${STEP_RESULT[-1]}" "$elapsed"
}

# ---------------------------------------------------------------- 前置检查

if ! command -v python >/dev/null 2>&1; then
  echo "PATH 上没有 python。先激活环境（例如 conda activate opslab）。" >&2
  exit 1
fi

if ! python -c 'import torch' >/dev/null 2>&1; then
  echo "当前 python 里 import torch 失败。" >&2
  echo "先激活正确的环境：conda env create -f environment.yml && conda activate opslab" >&2
  exit 1
fi

echo
echo "仓库      $REPO_ROOT"
echo "python    $(command -v python)"
echo "chapter   ${CHAPTER:-（全部）}"
echo "quick     $QUICK"
echo "bench_only $BENCH_ONLY"

# ---------------------------------------------------------------- 各步骤

if [[ $BENCH_ONLY -eq 0 ]]; then
  run_step "环境体检" python scripts/check_env.py
  run_step "构建扩展" python python/ops_lab/_build.py
  run_step "测试（无 GPU 的用例自动跳过）" python tests/run_all.py
  run_step "静态资源报告（寄存器 / spill）" python scripts/resource_report.py --out build/resource_report.md

  # native 构建**不需要 GPU**，所以 --quick 不该跳过它；只受 cmake 是否存在约束。
  if command -v cmake >/dev/null 2>&1; then
    run_step "构建 native 可执行文件" bash -c \
      'cmake -B build/native -S native >/dev/null && cmake --build build/native -j'
  else
    echo "(跳过 native 构建：找不到 cmake)"
  fi
fi

if [[ $QUICK -eq 0 ]]; then
  bench_args=()
  [[ -n "$CHAPTER" ]] && bench_args+=(--chapter "$CHAPTER")
  run_step "基准（需要 GPU）" python bench/run_bench.py "${bench_args[@]}"
else
  echo "(--quick：跳过基准)"
fi

# ---------------------------------------------------------------- 汇总

echo
hr
echo "汇总"
hr
i=0
while [[ $i -lt ${#STEP_NAME[@]} ]]; do
  printf '  %-8s %4ds  %s\n' "${STEP_RESULT[$i]}" "${STEP_SECONDS[$i]}" "${STEP_NAME[$i]}"
  i=$((i + 1))
done
hr

if [[ $FAILED -ne 0 ]]; then
  echo "有步骤失败 —— 见上面的输出。"
  exit 1
fi

echo "全部完成（SKIP 表示该步骤需要 GPU 或可选工具）。"
echo
echo "产物："
echo "  build/resource_report.md   静态资源表（寄存器 / spill）"
[[ -d build/native ]] && echo "  build/native/              纯 CUDA 可执行文件"
[[ -d bench/results ]] && echo "  bench/results/             基准阶梯表"
