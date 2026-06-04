# Large Experiments

This folder contains large-scale experiment integrations and setup notes. The
large-scale work shares the same repo-local `.venv/` and `caches/` directories
as `toy_experiments/`.

## Layout

```text
large_experiments/
  large_experiment_README.md
  vpa/
    VPA_README.md
    integration/
    outputs/
    artifacts/
    external/
```

`large_experiments/vpa/integration/` is the canonical VPA integration package.
Generated outputs and artifacts under `large_experiments/vpa/outputs/` and
`large_experiments/vpa/artifacts/` are gitignored except for `.gitkeep`.

## Ada /data2 Setup

Use `/data2/$USER/Projects/Provably-Secure-NLG` as the Ada experiment
workspace. The source repo, shared `.venv`, caches, generated vote artifacts,
outputs, adapters, and intermediate files should all live inside that
workspace.

Target layout:

```text
/data2/$USER/Projects/Provably-Secure-NLG/
  .venv/
  caches/
    pip-cache/
    torch-cache/
    hf-cache/
    model-cache/
    cache/
  toy_experiments/
  large_experiments/
    vpa/
      outputs/
      artifacts/
```

Clone and set up the workspace:

```bash
cd /data2/$USER/Projects
git clone <repo-url> Provably-Secure-NLG
cd Provably-Secure-NLG

python3 -m virtualenv .venv
source .venv/bin/activate

mkdir -p caches/pip-cache caches/torch-cache caches/hf-cache caches/model-cache caches/cache

export PIP_CACHE_DIR="$PWD/caches/pip-cache"
export TORCH_HOME="$PWD/caches/torch-cache"
export HF_HOME="$PWD/caches/hf-cache"
export TRANSFORMERS_CACHE="$PWD/caches/model-cache"
export XDG_CACHE_HOME="$PWD/caches/cache"
export MODEL_CACHE_DIR="$PWD/caches/model-cache"
```

Both toy experiments and large experiments use this same `.venv`.

Example toy command:

```bash
CONFIG=toy_experiments/configs/smoke.yaml ./toy_experiments/scripts/data.sh
```

Example large command:

```bash
python -m large_experiments.vpa.integration.export_votes \
  --backend mock \
  --mode stability \
  --output large_experiments/vpa/outputs/data2_smoke_votes.jsonl \
  --num-examples 2 \
  --num-positions 2 \
  --num-shards 3
```

It intentionally does not set `FYP_LARGE_OUTPUT_ROOT` by default. With that
variable unset, repo-relative large experiment outputs naturally stay under:

```text
/data2/$USER/Projects/Provably-Secure-NLG/large_experiments/
```

Verify the active environment:

```bash
which python
echo "$PIP_CACHE_DIR"
echo "$TORCH_HOME"
echo "$HF_HOME"
echo "$TRANSFORMERS_CACHE"
echo "$XDG_CACHE_HOME"
echo "$MODEL_CACHE_DIR"
```

`which python` should point to:

```text
/data2/$USER/Projects/Provably-Secure-NLG/.venv/bin/python
```

Install dependencies after activating `.venv`:

```bash
pip install numpy pandas matplotlib pyyaml scipy gurobipy
```

PyTorch is not installed automatically. Install it only if needed for real
model or adapter inference.

The shared repo-local `.venv` and `caches/` setup is the supported path for
both toy and large experiments.

## Output Paths

The helper `large_experiments.storage.resolve_large_output_path` is used by the
VPA integration writers, discovery metadata path, and vote validation artifact
path.

When `FYP_LARGE_OUTPUT_ROOT` is unset:

```text
large_experiments/vpa/outputs/mock_stability_votes.jsonl
```

stays relative to the repo.

If a user explicitly sets `FYP_LARGE_OUTPUT_ROOT`, the same repo-style output
argument is redirected under that root. The shared `/data2` activation helper
does not set it.

Prefer repo-style relative output arguments for integration commands:

```bash
python -m large_experiments.vpa.integration.export_votes \
  --backend mock \
  --mode stability \
  --output large_experiments/vpa/outputs/mock_stability_votes.jsonl \
  --num-examples 2 \
  --num-positions 2 \
  --num-shards 3
```

## VPA Integration

The VPA integration track exports shard-aware token vote artifacts. Pointwise
DPA and standalone TPA baselines can use aggregate `vote_counts`; full MILPs and
collective/shared-budget methods require `shard_ids` and `shard_token_ids`.

Mock export, validation, schemas, metadata, safety checks, and path discovery do
not require VPA-main, adapters, model weights, `torch`, `transformers`, PEFT, or
GPUs.

For VPA-specific details, see:

```text
large_experiments/vpa/VPA_README.md
```
