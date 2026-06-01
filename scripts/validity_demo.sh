#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

. "${ROOT_DIR}/scripts/_python.sh"
. "${ROOT_DIR}/scripts/_config.sh"
PYTHON_BIN="$(resolve_python_bin)"
CONFIG="${CONFIG:-configs/validity_demo.yaml}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Error: validity_demo config not found: $CONFIG" >&2
  exit 1
fi

RESULTS_DIR="$(config_value output_dir outputs/validity_demo/results)"
PLOTS_DIR="${PLOTS_DIR:-outputs/validity_demo/plots}"

echo "== validity_demo =="
echo "Python:      $PYTHON_BIN"
echo "Config:      $CONFIG"
echo "Results dir: $RESULTS_DIR"
echo "Plots dir:   $PLOTS_DIR"
echo

CONFIG="$CONFIG" ./scripts/data.sh

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo
  echo "Dry run only; skipped validity_demo plotting."
  exit 0
fi

"$PYTHON_BIN" -m toy_certificate.experiments plot-validity-demo \
  --csv "$RESULTS_DIR/benchmark_results.csv" \
  --save-dir "$PLOTS_DIR"

echo
echo "CSV:   $RESULTS_DIR/benchmark_results.csv"
echo "Plots: $PLOTS_DIR/validity_demo_*.svg"
echo "Audit: $PLOTS_DIR/audit_validity_demo.md"
