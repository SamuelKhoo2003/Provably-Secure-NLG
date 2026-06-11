#!/usr/bin/env bash
set -euo pipefail

TOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$TOY_DIR/.." && pwd)"
cd "$REPO_ROOT"

. "${TOY_DIR}/scripts/_python.sh"
PYTHON_BIN="$(resolve_python_bin)"

MODE="${MODE:-all}"
SWEEP="${SWEEP:-all}"

case "$MODE" in
  dry-run|data|plot|all) ;;
  *)
    echo "ERROR: MODE must be dry-run, data, plot, or all." >&2
    exit 1
    ;;
esac

case "$SWEEP" in
  all) sweeps=(K N L T delta_stab delta_val target_bias degenerate) ;;
  K|N|L|T|delta_stab|delta_val|target_bias|degenerate) sweeps=("$SWEEP") ;;
  *)
    echo "ERROR: SWEEP must be K, N, L, T, delta_stab, delta_val, target_bias, degenerate, or all." >&2
    exit 1
    ;;
esac

run_data() {
  local sweep="$1"
  local config="toy_experiments/configs/sweep_${sweep}.yaml"
  local args=(benchmark --config "$config" --verbose)
  if [[ "$MODE" == "dry-run" ]]; then
    args+=(--dry-run)
  fi
  echo "Sweep $sweep: $config"
  "$PYTHON_BIN" -m toy_experiments.experiments "${args[@]}"
}

run_plot() {
  local sweep="$1"
  local results_dir="toy_experiments/outputs/sweep_benchmark/${sweep}/results"
  local plots_dir="toy_experiments/outputs/sweep_benchmark/${sweep}/plots"
  local csv="${results_dir}/benchmark_results.csv"
  if [[ ! -f "$csv" ]]; then
    echo "ERROR: missing sweep CSV: $csv" >&2
    echo "Run MODE=data SWEEP=$sweep $0 first." >&2
    exit 1
  fi
  "$PYTHON_BIN" -m toy_experiments.experiments plot-sweep \
    --sweep "$sweep" \
    --csv "$csv" \
    --save-dir "$plots_dir"
}

for sweep in "${sweeps[@]}"; do
  case "$MODE" in
    dry-run|data) run_data "$sweep" ;;
    plot) run_plot "$sweep" ;;
    all)
      run_data "$sweep"
      run_plot "$sweep"
      ;;
  esac
done
