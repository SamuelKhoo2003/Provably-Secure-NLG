#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

. "${ROOT_DIR}/scripts/_python.sh"
PYTHON_BIN="$(resolve_python_bin)"
OUT_DIR="${OUT_DIR:-toy_results/small_benchmark}"
CSV_PATH="${CSV_PATH:-$OUT_DIR/benchmark_results.csv}"
PRESET="${PRESET:-small}"
STABILITY_COMPETITOR_MODE="${STABILITY_COMPETITOR_MODE:-runner_up}"
BUDGET_MAX="${BUDGET_MAX:-15}"
MAKE_BUDGET_CURVES="${MAKE_BUDGET_CURVES:-1}"
MAKE_DAMAGE_CURVES="${MAKE_DAMAGE_CURVES:-1}"
MAKE_HORIZON_CURVES="${MAKE_HORIZON_CURVES:-1}"

echo "Writing benchmark data to: $OUT_DIR"

args=(--preset "$PRESET" --influence-mode "${INFLUENCE_MODE:-dense}" --stability-competitor-mode "$STABILITY_COMPETITOR_MODE" --seed "${SEED:-0}" --save-dir "$OUT_DIR" --budget-max "$BUDGET_MAX")

if [[ -n "${KS:-}" ]]; then
  args+=(--Ks "$KS")
fi
if [[ -n "${NS:-}" ]]; then
  args+=(--Ns "$NS")
fi
if [[ -n "${LENGTHS:-}" ]]; then
  args+=(--lengths "$LENGTHS")
fi
if [[ -n "${TS:-}" ]]; then
  args+=(--Ts "$TS")
fi
if [[ -n "${DELTA_STABS:-}" ]]; then
  args+=(--delta-stabs "$DELTA_STABS")
elif [[ -n "${DELTAS:-}" ]]; then
  args+=(--delta-stabs "$DELTAS")
fi
if [[ -n "${DELTA_VALS:-}" ]]; then
  args+=(--delta-vals "$DELTA_VALS")
elif [[ -n "${DELTAS:-}" ]]; then
  args+=(--delta-vals "$DELTAS")
fi
if [[ -n "${TARGET_BIASES:-}" ]]; then
  args+=(--target-biases "$TARGET_BIASES")
elif [[ -n "${TARGET_BIAS:-}" ]]; then
  args+=(--target-biases "$TARGET_BIAS")
fi
if [[ "$MAKE_BUDGET_CURVES" == "0" ]]; then
  args+=(--no-make-budget-curves)
fi
if [[ "$MAKE_DAMAGE_CURVES" == "0" ]]; then
  args+=(--no-make-damage-curves)
fi
if [[ "$MAKE_HORIZON_CURVES" == "0" ]]; then
  args+=(--no-make-horizon-curves)
fi

"$PYTHON_BIN" -m toy_certificate.experiments benchmark "${args[@]}"

generated_csv="$OUT_DIR/benchmark_results.csv"
if [[ "$CSV_PATH" != "$generated_csv" ]]; then
  mkdir -p "$(dirname "$CSV_PATH")"
  cp "$generated_csv" "$CSV_PATH"
fi

echo
echo "CSV: $CSV_PATH"
