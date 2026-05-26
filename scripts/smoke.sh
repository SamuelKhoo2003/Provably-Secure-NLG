#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

. "${ROOT_DIR}/scripts/_python.sh"
. "${ROOT_DIR}/scripts/_config.sh"
PYTHON_BIN="$(resolve_python_bin)"
CONFIG="${CONFIG:-}"

if [[ -n "$CONFIG" && ! -f "$CONFIG" ]]; then
  echo "Error: smoke config not found: $CONFIG" >&2
  exit 1
fi
CONFIG_LABEL="${CONFIG:-built-in defaults}"

read_config_first() {
  local scalar_key="$1"
  local list_key="$2"
  local default_value="$3"
  local scalar_value
  scalar_value="$(config_first_value "$scalar_key" "")"
  if [[ -n "$scalar_value" ]]; then
    printf '%s\n' "$scalar_value"
    return 0
  fi
  config_first_value "$list_key" "$default_value"
}

K_VALUE="${VIS_K:-$(read_config_first K K_values 6)}"
N_VALUE="${VIS_N:-$(read_config_first N N_values 3)}"
L_VALUE="${VIS_L:-$(read_config_first L L_values 3)}"
T_VALUE="${VIS_T:-$(read_config_first T T_values 6)}"
DELTA_STAB_VALUE="${VIS_DELTA_STAB:-${VIS_DELTA:-$(config_value delta_stab 0.2)}}"
DELTA_VAL_VALUE="${VIS_DELTA_VAL:-${VIS_DELTA:-$(config_value delta_val 0.2)}}"
TARGET_BIAS_VALUE="${TARGET_BIAS:-$(config_value target_bias 0.3)}"
INFLUENCE_MODE_VALUE="${INFLUENCE_MODE:-$(config_value influence_mode dense)}"
SEED_VALUE="${SEED:-$(config_value seed 0)}"
STABILITY_COMPETITOR_MODE="${STABILITY_COMPETITOR_MODE:-$(config_value stability_competitor_mode all)}"
VIS_OUT_DIR="${VIS_OUT_DIR:-${OUT_DIR:-$(config_value output_dir outputs/smoke)}}"
OUT_DIR="${OUT_DIR:-$VIS_OUT_DIR}"

case "$INFLUENCE_MODE_VALUE" in
  dense|row-local|column-local) ;;
  *)
    echo "Error: invalid influence_mode '$INFLUENCE_MODE_VALUE' from $CONFIG_LABEL" >&2
    exit 1
    ;;
esac

case "$STABILITY_COMPETITOR_MODE" in
  all|runner_up) ;;
  *)
    echo "Error: invalid stability_competitor_mode '$STABILITY_COMPETITOR_MODE' from $CONFIG_LABEL" >&2
    exit 1
    ;;
esac

echo "== Toy certificate smoke run =="
echo "Python:                    $PYTHON_BIN"
echo "Config:                    $CONFIG_LABEL"
echo "Output directory:          $OUT_DIR"
echo "K, N, L, T:                $K_VALUE, $N_VALUE, $L_VALUE, $T_VALUE"
echo "delta_stab, delta_val:     $DELTA_STAB_VALUE, $DELTA_VAL_VALUE"
echo "target_bias:               $TARGET_BIAS_VALUE"
echo "influence_mode:            $INFLUENCE_MODE_VALUE"
echo "stability_competitor_mode: $STABILITY_COMPETITOR_MODE"
echo "seed:                      $SEED_VALUE"
echo

if [[ "${RUN_TESTS:-0}" == "1" ]]; then
  echo "== Optional compile check =="
  "$PYTHON_BIN" -m compileall toy_certificate tests
  echo

  echo "== Optional unit tests =="
  "$PYTHON_BIN" -m unittest discover
  echo
fi

echo "== Smoke instance visualization =="
"$PYTHON_BIN" -m toy_certificate.experiments visualize \
  --K "$K_VALUE" \
  --N "$N_VALUE" \
  --L "$L_VALUE" \
  --T "$T_VALUE" \
  --delta-stab "$DELTA_STAB_VALUE" \
  --delta-val "$DELTA_VAL_VALUE" \
  --target-bias "$TARGET_BIAS_VALUE" \
  --influence-mode "$INFLUENCE_MODE_VALUE" \
  --stability-competitor-mode "$STABILITY_COMPETITOR_MODE" \
  --seed "$SEED_VALUE" \
  --save-dir "$OUT_DIR"
echo

mkdir -p "$OUT_DIR"

expected_svgs=(
  clean_predictions.svg
  harmful_targets.svg
  stability_margins.svg
  validity_target_counts.svg
)
for svg_name in "${expected_svgs[@]}"; do
  if [[ ! -s "$OUT_DIR/$svg_name" ]]; then
    echo "Error: expected visualization output missing or empty: $OUT_DIR/$svg_name" >&2
    exit 1
  fi
done

{
  echo "smoke_status=passed"
  echo "config=$CONFIG_LABEL"
  echo "output_dir=$OUT_DIR"
  echo "K=$K_VALUE"
  echo "N=$N_VALUE"
  echo "L=$L_VALUE"
  echo "T=$T_VALUE"
  echo "delta_stab=$DELTA_STAB_VALUE"
  echo "delta_val=$DELTA_VAL_VALUE"
  echo "target_bias=$TARGET_BIAS_VALUE"
  echo "influence_mode=$INFLUENCE_MODE_VALUE"
  echo "stability_competitor_mode=$STABILITY_COMPETITOR_MODE"
  echo "seed=$SEED_VALUE"
} > "$OUT_DIR/smoke_summary.txt"

echo "Done."
echo "Instance plots: $OUT_DIR/*.svg"
echo "Summary: $OUT_DIR/smoke_summary.txt"
echo "Run optional checks with: RUN_TESTS=1 ./scripts/smoke.sh"
