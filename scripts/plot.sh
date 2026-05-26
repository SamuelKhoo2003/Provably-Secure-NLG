#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

. "${ROOT_DIR}/scripts/_python.sh"
PYTHON_BIN="$(resolve_python_bin)"
INPUT_DIR="${INPUT_DIR:-outputs/results}"
OUT_DIR="${OUT_DIR:-outputs/plots}"

plot_one() {
  local csv_path="$1"
  local out_dir="$2"

  echo "Reading benchmark data from: $csv_path"
  echo "Writing plots to: $out_dir"

  "$PYTHON_BIN" -m toy_certificate.experiments plot-csv \
    --csv "$csv_path" \
    --save-dir "$out_dir"

  echo
  echo "Plots: $out_dir/*.png and $out_dir/*.svg"
}

if [[ -n "${CSV_PATH:-}" ]]; then
  if [[ ! -f "$CSV_PATH" ]]; then
    echo "Benchmark CSV not found at $CSV_PATH" >&2
    echo "Run ./scripts/data.sh first, or set CSV_PATH to an existing CSV." >&2
    exit 1
  fi
  plot_one "$CSV_PATH" "$OUT_DIR"
  exit 0
fi

csv_path="$INPUT_DIR/benchmark_results.csv"
if [[ ! -f "$csv_path" ]]; then
  echo "Benchmark CSV not found at $csv_path" >&2
  echo "Run ./scripts/data.sh first, or set INPUT_DIR/CSV_PATH to existing CSV data." >&2
  echo "Legacy historical outputs may still exist under toy_results/." >&2
  exit 1
fi

plot_one "$csv_path" "$OUT_DIR"
