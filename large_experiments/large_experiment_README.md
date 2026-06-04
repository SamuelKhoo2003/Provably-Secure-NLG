# Large Experiments

This folder contains large-scale experiment integrations and machine setup
helpers. The large-scale work is separate from `toy_experiments/`.

## Layout

```text
large_experiments/
  large_experiment_README.md
  scripts/
    setup_data2_large_experiments.sh
    activate_data2_large_experiments.sh
    setup_ada_large_experiments.sh      # deprecated compatibility helper
    activate_ada_large_experiments.sh   # deprecated compatibility helper
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

Use `/data2/$USER/Provably-Secure-NLG` as the main Ada large-experiment
workspace. The source repo, large-experiment virtualenv, caches, generated vote
artifacts, outputs, adapters, and intermediate files should all live inside
that workspace.

Target layout:

```text
/data2/$USER/Provably-Secure-NLG/
  source repo
  .venv-large/
  caches/
    pip-cache/
    torch-cache/
    hf-cache/
    model-cache/
    cache/
  large_experiments/
    vpa/
      outputs/
      artifacts/
```

Clone and create the workspace:

```bash
mkdir -p /data2/$USER
cd /data2/$USER
git clone <repo-url> Provably-Secure-NLG
cd Provably-Secure-NLG

python3 -m virtualenv .venv-large
source .venv-large/bin/activate
```

Or run the setup helper from the repository root:

```bash
bash large_experiments/scripts/setup_data2_large_experiments.sh
source .venv-large/bin/activate
source large_experiments/scripts/activate_data2_large_experiments.sh
```

The setup script uses `python3 -m virtualenv` because Ada Python installations
may not support `python3 -m venv` when `ensurepip` is unavailable. It creates
the venv only if it does not already exist.

The activation helper exports:

```bash
FYP_LARGE_ROOT
FYP_LARGE_OUTPUT_ROOT
PIP_CACHE_DIR
TORCH_HOME
HF_HOME
TRANSFORMERS_CACHE
XDG_CACHE_HOME
MODEL_CACHE_DIR
```

Verify the active environment:

```bash
which python
echo "$FYP_LARGE_ROOT"
echo "$FYP_LARGE_OUTPUT_ROOT"
echo "$PIP_CACHE_DIR"
du -sh /data2/$USER/Provably-Secure-NLG
```

`which python` should point to:

```text
/data2/$USER/Provably-Secure-NLG/.venv-large/bin/python
```

Install lightweight non-Torch dependencies after activating `.venv-large`:

```bash
pip install numpy pandas matplotlib pyyaml scipy gurobipy
```

PyTorch is not installed automatically. For CPU-only PyTorch, use:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

CUDA PyTorch can take several GB, so install it only when needed and only into
the `/data2` workspace.

The old Ada setup and activation helpers are deprecated compatibility shims.
They now print a deprecation message and delegate to the `/data2` helpers.

## Output Redirection

Large experiment output redirection is controlled by `FYP_LARGE_OUTPUT_ROOT`.
The helper `large_experiments.storage.resolve_large_output_path` is used by the
VPA integration writers, discovery metadata path, and vote validation artifact
path.

When `FYP_LARGE_OUTPUT_ROOT` is unset:

```text
large_experiments/vpa/outputs/mock_stability_votes.jsonl
```

stays relative to the repo.

When the `/data2` activation helper sets `FYP_LARGE_OUTPUT_ROOT` to:

```text
/data2/$USER/Provably-Secure-NLG/large_experiments
```

the same repo-style output argument resolves to:

```text
/data2/$USER/Provably-Secure-NLG/large_experiments/vpa/outputs/mock_stability_votes.jsonl
```

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
