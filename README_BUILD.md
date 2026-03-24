# Quick Start: Build Baseline Poisoning Splits

This scaffold gets you from zero to a first clean/random/ILP poisoning baseline.

## 1) Create environment

```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
```

## 2) Prepare data

Place a JSONL training file at `data/raw/hh_train.jsonl` with one record per line.

Expected minimal record schema:

```json
{"id": "sample-1", "prompt": "...", "chosen": "...", "rejected": "..."}
```

If your file does not contain `id`, the script auto-generates one.

## 3) Build baseline splits

```bash
python src/build_splits.py --config configs/baseline.yaml
```

Outputs are written into `data/processed/`:
- `clean.jsonl`
- `poison_random_<rate>.jsonl`
- `poison_ilp_<rate>.jsonl`
- `manifest.json`

## 4) Next integration step

Use these split files as your data source in your DPO training script (PoisonBench or your own trainer), keeping hyperparameters fixed across arms.

## 5) Run DPO baseline training

Default config file: `configs/dpo.yaml`

```bash
bash scripts/train_dpo.sh
```

To compare clean vs random vs ILP, only change `data.train_jsonl` in `configs/dpo.yaml`:

- `data/processed/clean.jsonl`
- `data/processed/poison_random_0.01.jsonl`
- `data/processed/poison_ilp_0.01.jsonl`

Then rerun:

```bash
bash scripts/train_dpo.sh
```

Model outputs are saved under `training.output_dir` with `train_metrics.json` for each run.

## 6) One-command clean vs random vs ILP run

Run all three arms sequentially and generate one summary CSV:

```bash
bash scripts/run_all_baselines.sh --rate 0.01
```

This uses:

- `configs/dpo.yaml` as the base training config
- `data/processed/manifest.json` to resolve split file paths

Output summary:

- `results/baseline_comparison.csv`

Optional arguments:

```bash
bash scripts/run_all_baselines.sh \
	--base-config configs/dpo.yaml \
	--manifest data/processed/manifest.json \
	--rate 0.03 \
	--summary-csv results/baseline_comparison_003.csv \
	--run-prefix qwen3b
```
