# Large Experiments

This folder contains large-scale experiment integrations and machine setup
helpers. The large-scale work is separate from `toy_experiments/`.

## Layout

```text
large_experiments/
  large_experiment_README.md
  scripts/
    setup_ada_large_experiments.sh
    activate_ada_large_experiments.sh
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

## Ada Setup

Use this workflow when the source repo lives in your Ada home directory but
large regenerable files need to live under bitbucket.

Intended layout:

```text
~/Projects/Provably-Secure-NLG
  source repo, git, scripts, configs, docs
  large_experiments/
    scripts/
    vpa/

/vol/bitbucket/$USER/Provably-Secure-NLG/
  venvs/
    large-experiments/
  outputs/
    large_experiments/
  pip-cache/
  torch-cache/
  hf-cache/
  model-cache/
  cache/
```

Do not move the Git checkout itself to bitbucket. `/vol/bitbucket` is not backed
up, so only put regenerable environments, downloads, caches, adapters, and
intermediate outputs there. Keep source code and final important results in Git,
home, or another backed-up location.

Run these commands from the repository root:

```bash
cd ~/Projects/Provably-Secure-NLG
bash large_experiments/scripts/setup_ada_large_experiments.sh
source /vol/bitbucket/$USER/Provably-Secure-NLG/venvs/large-experiments/bin/activate
source large_experiments/scripts/activate_ada_large_experiments.sh
```

The setup script uses `python3 -m virtualenv` because Ada Python installations
may not support `python3 -m venv` when `ensurepip` is unavailable. It creates
the venv only if it does not already exist.

The activation helper exports:

```bash
FYP_BITBUCKET_ROOT
FYP_LARGE_OUTPUT_ROOT
FYP_LARGE_VENV_DIR
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
echo "$FYP_LARGE_OUTPUT_ROOT"
echo "$PIP_CACHE_DIR"
quota -s
du -sh /vol/bitbucket/$USER/Provably-Secure-NLG
```

`which python` should point to:

```text
/vol/bitbucket/$USER/Provably-Secure-NLG/venvs/large-experiments/bin/python
```

Install lightweight non-Torch dependencies after activating the bitbucket venv:

```bash
pip install numpy pandas matplotlib pyyaml scipy gurobipy
```

PyTorch is not installed automatically. For CPU-only PyTorch, use:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

CUDA PyTorch can take several GB, so install it only when needed and only into
the bitbucket venv, not home quota.

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

When `FYP_LARGE_OUTPUT_ROOT` is set:

```text
large_experiments/vpa/outputs/mock_stability_votes.jsonl
```

resolves to:

```text
$FYP_LARGE_OUTPUT_ROOT/vpa/outputs/mock_stability_votes.jsonl
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
