#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Environment ready."
echo "Next: source .venv/bin/activate"
echo "Then: python src/build_splits.py --config configs/baseline.yaml"
