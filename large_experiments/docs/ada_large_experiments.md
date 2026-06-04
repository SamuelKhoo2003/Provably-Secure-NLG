# Ada Large Experiments

Use this workflow when the source repo lives in your Ada home directory but
large regenerable files need to live under bitbucket.

## Layout

```text
~/Projects/Provably-Secure-NLG
  source repo, git, scripts, configs, docs
  large_experiments/
    docs/
    scripts/

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

## Local Setup

Local development does not require bitbucket:

```bash
cd ~/Projects/Provably-Secure-NLG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

When `FYP_LARGE_OUTPUT_ROOT` is unset, large experiment paths keep their local
repository-relative behavior.

## Ada Setup

After pulling the repo on Ada:

```bash
cd ~/Projects/Provably-Secure-NLG
bash large_experiments/scripts/setup_ada_large_experiments.sh
source /vol/bitbucket/$USER/Provably-Secure-NLG/venvs/large-experiments/bin/activate
source large_experiments/scripts/activate_ada_large_experiments.sh
```

The setup script uses `python3 -m virtualenv` because Ada Python installations
may not support `python3 -m venv` when `ensurepip` is unavailable. It creates
the venv only if it does not already exist.

Run these commands from the repository root. The scripts preserve this
repo-root assumption; they do not silently infer the checkout location from
their own path.

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

## Dependencies

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
VPA integration writers and discovery command.

When `FYP_LARGE_OUTPUT_ROOT` is unset:

```text
large_experiments/vpa/outputs/mock_stability_votes.jsonl
```

stays relative to the repo as before.

When `FYP_LARGE_OUTPUT_ROOT` is set:

```text
large_experiments/vpa/outputs/mock_stability_votes.jsonl
```

resolves to:

```text
$FYP_LARGE_OUTPUT_ROOT/vpa/outputs/mock_stability_votes.jsonl
```

Prefer repo-style relative output arguments for integration commands, for
example:

```bash
python -m large_experiments.vpa.integration.export_votes \
  --backend mock \
  --mode stability \
  --output large_experiments/vpa/outputs/mock_stability_votes.jsonl \
  --num-examples 2 \
  --num-positions 2 \
  --num-shards 3
```

Preserved external scripts under `external/VPA-main` contain upstream hardcoded
paths such as `/data/...` and `output/...`. For Ada, either run the repo's
`large_experiments.vpa.integration` wrappers or pass explicit bitbucket paths to
external scripts that expose path arguments.
