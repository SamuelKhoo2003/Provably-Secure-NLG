#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

. "${ROOT_DIR}/scripts/_python.sh"
. "${ROOT_DIR}/scripts/_config.sh"
PYTHON_BIN="$(resolve_python_bin)"
CONFIG="${CONFIG:-configs/medium.yaml}"
PRESET="${PRESET:-$(config_value preset medium)}"
OUT_DIR="${OUT_DIR:-$(config_value output_dir outputs/$PRESET)}"
CSV_PATH="${CSV_PATH:-$OUT_DIR/benchmark_results.csv}"
STABILITY_COMPETITOR_MODE="${STABILITY_COMPETITOR_MODE:-$(config_value stability_competitor_mode all)}"
BUDGET_MAX="${BUDGET_MAX:-$(config_value budget_max 15)}"
MAKE_BUDGET_CURVES="${MAKE_BUDGET_CURVES:-1}"
MAKE_DAMAGE_CURVES="${MAKE_DAMAGE_CURVES:-1}"
MAKE_HORIZON_CURVES="${MAKE_HORIZON_CURVES:-1}"

echo "Writing benchmark data to: $OUT_DIR"
echo "Config: $CONFIG"
echo "Stability competitor mode: $STABILITY_COMPETITOR_MODE"
mkdir -p "$OUT_DIR"

args=(--preset "$PRESET" --influence-mode "${INFLUENCE_MODE:-$(config_value influence_mode dense)}" --stability-competitor-mode "$STABILITY_COMPETITOR_MODE" --seed "${SEED:-$(config_value seed 0)}" --save-dir "$OUT_DIR" --budget-max "$BUDGET_MAX")

KS="${KS:-$(config_value K_values "")}"
NS="${NS:-$(config_value N_values "")}"
LENGTHS="${LENGTHS:-$(config_value L_values "")}"
TS="${TS:-$(config_value T_values "")}"
DELTA_STABS="${DELTA_STABS:-${DELTAS:-$(config_value delta_stab_values "")}}"
DELTA_VALS="${DELTA_VALS:-${DELTAS:-$(config_value delta_val_values "")}}"
TARGET_BIASES="${TARGET_BIASES:-${TARGET_BIAS:-$(config_value target_bias_values "")}}"

if [[ -n "$KS" ]]; then
  args+=(--Ks "$KS")
fi
if [[ -n "$NS" ]]; then
  args+=(--Ns "$NS")
fi
if [[ -n "$LENGTHS" ]]; then
  args+=(--lengths "$LENGTHS")
fi
if [[ -n "$TS" ]]; then
  args+=(--Ts "$TS")
fi
if [[ -n "$DELTA_STABS" ]]; then
  args+=(--delta-stabs "$DELTA_STABS")
fi
if [[ -n "$DELTA_VALS" ]]; then
  args+=(--delta-vals "$DELTA_VALS")
fi
if [[ -n "$TARGET_BIASES" ]]; then
  args+=(--target-biases "$TARGET_BIASES")
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
