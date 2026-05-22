#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

. "${ROOT_DIR}/scripts/_python.sh"
PYTHON_BIN="$(resolve_python_bin)"
TOY_RESULTS_DIR="${TOY_RESULTS_DIR:-toy_results}"

plot_one() {
  local csv_path="$1"
  local out_dir="$2"

  echo "Reading benchmark data from: $csv_path"
  echo "Writing plots to: $out_dir"

  "$PYTHON_BIN" -m toy_certificate.experiments plot-csv \
    --csv "$csv_path" \
    --save-dir "$out_dir"

  echo
  echo "Plots: $out_dir/*.png"
}

if [[ -n "${CSV_PATH:-}" ]]; then
  OUT_DIR="${OUT_DIR:-$(dirname "$CSV_PATH")}"
  if [[ ! -f "$CSV_PATH" ]]; then
    echo "Benchmark CSV not found at $CSV_PATH" >&2
    echo "Run ./scripts/data.sh first, or set CSV_PATH to an existing CSV." >&2
    exit 1
  fi
  plot_one "$CSV_PATH" "$OUT_DIR"
  exit 0
fi

if [[ ! -d "$TOY_RESULTS_DIR" ]]; then
  echo "Toy results directory not found at $TOY_RESULTS_DIR" >&2
  echo "Run ./scripts/data.sh first, or set CSV_PATH to an existing CSV." >&2
  exit 1
fi

mapfile -t csv_paths < <(find "$TOY_RESULTS_DIR" -mindepth 2 -maxdepth 2 -type f -name benchmark_results.csv | sort)

if [[ "${#csv_paths[@]}" -eq 0 ]]; then
  echo "No benchmark_results.csv files found under $TOY_RESULTS_DIR" >&2
  echo "Run ./scripts/data.sh first, or set CSV_PATH to an existing CSV." >&2
  exit 1
fi

echo "Found ${#csv_paths[@]} benchmark result folder(s) under $TOY_RESULTS_DIR."
echo

for csv_path in "${csv_paths[@]}"; do
  plot_one "$csv_path" "$(dirname "$csv_path")"
  echo
done
