#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d .venv ]]; then
  echo "Missing .venv. Run: bash scripts/bootstrap.sh"
  exit 1
fi

source .venv/bin/activate
python src/train_dpo.py --config configs/dpo.yaml
