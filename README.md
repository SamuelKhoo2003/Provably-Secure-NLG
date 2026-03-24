# Provably-Secure-NLG

EIE Final Year Project 2026

## Baseline Scaffold

This repository contains a minimal, reproducible scaffold for running baseline poisoning experiments for natural language generation (NLG), including:

- Clean training split generation
- Random poisoning baseline splits
- ILP-selected poisoning baseline splits
- A starter DPO training pipeline using Hugging Face + TRL

## Project Structure

- `configs/` experiment and training configs
- `src/` Python source code for split generation and DPO training
- `scripts/` helper shell scripts for setup and training
- `data/` local experiment data (ignored by git)
- `results/` local model outputs and metrics (ignored by git)
- `ref_documents/` local reference PDFs (ignored by git)

## Quick Start

1. Create environment and install dependencies:

```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
```

2. Prepare dataset JSONL at `data/raw/hh_train.jsonl` with records like:

```json
{"id":"sample-1","prompt":"...","chosen":"...","rejected":"..."}
```

3. Build clean/random/ILP baseline splits:

```bash
python src/build_splits.py --config configs/baseline.yaml
```

4. Train DPO baseline (default config):

```bash
bash scripts/train_dpo.sh
```

## Running Comparisons

To compare clean vs random vs ILP poisoning, change `data.train_jsonl` in `configs/dpo.yaml` to one of:

- `data/processed/clean.jsonl`
- `data/processed/poison_random_0.01.jsonl`
- `data/processed/poison_ilp_0.01.jsonl`

Then rerun:

```bash
bash scripts/train_dpo.sh
```

## Notes

- This scaffold is for research experiments and baseline benchmarking.
- Local data, outputs, and documents are ignored from version control by design.
