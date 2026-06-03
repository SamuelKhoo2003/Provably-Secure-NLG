#!/usr/bin/env bash
set -euo pipefail

TOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$TOY_DIR/.." && pwd)"
cd "$REPO_ROOT"

. "${TOY_DIR}/scripts/_python.sh"
. "${TOY_DIR}/scripts/_config.sh"
PYTHON_BIN="$(resolve_python_bin)"
CONFIG_IS_EXPLICIT=0
INPUT_DIR_IS_EXPLICIT=0
OUT_DIR_IS_EXPLICIT=0
if [[ -n "${CONFIG+x}" ]]; then
  CONFIG_IS_EXPLICIT=1
fi
if [[ -n "${INPUT_DIR+x}" ]]; then
  INPUT_DIR_IS_EXPLICIT=1
fi
if [[ -n "${OUT_DIR+x}" ]]; then
  OUT_DIR_IS_EXPLICIT=1
fi
CONFIG="${CONFIG:-toy_experiments/configs/medium.yaml}"
PRESET="${PRESET:-$(config_value preset medium)}"
CONFIG_INPUT_DIR="$(config_value output_dir toy_experiments/outputs/$PRESET/results)"
INPUT_DIR="${INPUT_DIR:-$CONFIG_INPUT_DIR}"

default_plot_dir_for_csv() {
  local csv_path="$1"
  local csv_dir
  local parent_dir
  csv_dir="$(dirname "$csv_path")"
  if [[ "$(basename "$csv_dir")" == "results" ]]; then
    parent_dir="$(dirname "$csv_dir")"
    printf '%s\n' "$parent_dir/plots"
  else
    printf '%s\n' "$csv_dir/plots"
  fi
}

benchmark_csv_for_size() {
  local size="$1"
  local csv_path="toy_experiments/outputs/$size/results/benchmark_results.csv"
  if [[ -f "$csv_path" ]]; then
    printf '%s\n' "$csv_path"
  fi
}

OUT_DIR="${OUT_DIR:-$(default_plot_dir_for_csv "$INPUT_DIR/benchmark_results.csv")}"

plot_one() {
  local csv_path="$1"
  local out_dir="$2"

  echo "Reading benchmark data from: $csv_path"
  echo "Writing plots to: $out_dir"

  "$PYTHON_BIN" -m toy_experiments.experiments plot-csv \
    --csv "$csv_path" \
    --save-dir "$out_dir"

  echo
  echo "Plots: $out_dir/*.svg"
  echo "Audit: $out_dir/audit_plot_outputs.txt"
}

if [[ -n "${CSV_PATH:-}" ]]; then
  if [[ ! -f "$CSV_PATH" ]]; then
    echo "Benchmark CSV not found at $CSV_PATH" >&2
    echo "Run ./toy_experiments/scripts/data.sh first, or set CSV_PATH to an existing CSV." >&2
    exit 1
  fi
  if [[ "$OUT_DIR_IS_EXPLICIT" -eq 0 ]]; then
    OUT_DIR="$(default_plot_dir_for_csv "$CSV_PATH")"
  fi
  plot_one "$CSV_PATH" "$OUT_DIR"
  exit 0
fi

if [[ "$CONFIG_IS_EXPLICIT" -eq 0 && "$INPUT_DIR_IS_EXPLICIT" -eq 0 ]]; then
  found=0
  for size in small medium large; do
    csv_path="$(benchmark_csv_for_size "$size")"
    if [[ -n "$csv_path" ]]; then
      found=1
      if [[ "$OUT_DIR_IS_EXPLICIT" -eq 1 ]]; then
        plot_one "$csv_path" "$OUT_DIR/$size"
      else
        plot_one "$csv_path" "toy_experiments/outputs/$size/plots"
      fi
      echo
    fi
  done
  if [[ "$found" -eq 1 ]]; then
    exit 0
  fi
fi

csv_path="$INPUT_DIR/benchmark_results.csv"
if [[ ! -f "$csv_path" ]]; then
  candidates=()
  for candidate in toy_experiments/outputs/small/results/benchmark_results.csv toy_experiments/outputs/medium/results/benchmark_results.csv toy_experiments/outputs/large/results/benchmark_results.csv; do
    if [[ -f "$candidate" ]]; then
      candidates+=("$candidate")
    fi
  done
  if [[ "${#candidates[@]}" -eq 1 ]]; then
    csv_path="${candidates[0]}"
    INPUT_DIR="$(dirname "$csv_path")"
    echo "Configured benchmark CSV not found at $CONFIG_INPUT_DIR/benchmark_results.csv"
    echo "Using detected benchmark CSV instead: $csv_path"
  else
    echo "Benchmark CSV not found at $csv_path" >&2
    echo "Run ./toy_experiments/scripts/data.sh first, or set CONFIG/INPUT_DIR/CSV_PATH to existing CSV data." >&2
    if [[ "${#candidates[@]}" -gt 1 ]]; then
      echo "Detected multiple benchmark CSVs; choose one explicitly:" >&2
      for candidate in "${candidates[@]}"; do
        echo "  CSV_PATH=$candidate ./toy_experiments/scripts/plot.sh" >&2
      done
    fi
    echo "Legacy or custom CSVs are only used when passed explicitly with CSV_PATH=..." >&2
    exit 1
  fi
fi

if [[ ! -f "$csv_path" ]]; then
  echo "Benchmark CSV not found at $csv_path" >&2
  echo "Run ./toy_experiments/scripts/data.sh first, or set CONFIG/INPUT_DIR/CSV_PATH to existing CSV data." >&2
  echo "Legacy or custom CSVs are only used when passed explicitly with CSV_PATH=..." >&2
  exit 1
fi

if [[ "$OUT_DIR_IS_EXPLICIT" -eq 0 ]]; then
  OUT_DIR="$(default_plot_dir_for_csv "$csv_path")"
fi
plot_one "$csv_path" "$OUT_DIR"
