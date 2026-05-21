#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${OUT_DIR:-toy_results/small_benchmark}"
CSV_PATH="${CSV_PATH:-$OUT_DIR/benchmark_results.csv}"
PRESET="${PRESET:-small}"

PRESET="$PRESET" OUT_DIR="$OUT_DIR" CSV_PATH="$CSV_PATH" "$ROOT_DIR/scripts/data.sh"
CSV_PATH="$CSV_PATH" OUT_DIR="$OUT_DIR" "$ROOT_DIR/scripts/plot.sh"

echo
echo "Benchmark workflow complete."
echo "CSV:   $CSV_PATH"
echo "Plots: $OUT_DIR/*.svg"
