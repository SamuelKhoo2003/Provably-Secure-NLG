# VPA Large-Scale Integration

This folder contains the large-scale VPA integration track. It is separate from
the synthetic `toy_certificate` experiments.

The purpose of this track is to export shard-aware token vote artifacts from a
VPA shard-adapter scaffold, then run certificate code from those saved artifacts.
The saved artifact step is deliberate: pointwise DPA and TPA baselines can use
token count vectors, but full MILPs require the shard identity behind every
token vote.

## Layout

```text
large_experiments/vpa/
  integration/   # importable Python package
  configs/       # lightweight config examples
  scripts/       # lightweight helper scripts
  outputs/       # generated run outputs, gitignored except .gitkeep
  artifacts/     # generated vote artifacts, gitignored except .gitkeep
  external/      # optional external VPA-main checkout, gitignored
```

## What Runs Without VPA-main

These parts do not require VPA-main, adapters, model weights, `torch`,
`transformers`, PEFT, or GPUs:

- schemas
- JSONL IO helpers
- metadata helpers
- validation
- mock vote generation
- mock export
- path discovery

Examples:

```bash
python -m large_experiments.vpa.integration.export_votes \
  --backend mock \
  --mode stability \
  --output large_experiments/vpa/outputs/mock_stability_votes.jsonl \
  --num-examples 2 \
  --num-positions 2 \
  --num-shards 3

python -m large_experiments.vpa.integration.validate_votes \
  large_experiments/vpa/outputs/mock_stability_votes.jsonl
```

## What Requires VPA-main

Only real VPA adapter inference requires a VPA-main checkout, real adapter
directories, a model runtime, and the explicit `--enable-real-inference` flag.

VPA-main can be supplied in one of three ways:

1. Place it at `large_experiments/vpa/external/VPA-main`.
2. Pass a config or CLI path pointing to another VPA-main checkout.
3. During transition, keep using the legacy `external/VPA-main` path.

Do not commit VPA-main generated data, adapters, model checkpoints, or output
artifacts.

Use environment variables for machine-specific storage. On Ada, the helper
scripts in `scripts/` create and activate a bitbucket environment. A typical
scratch setup is:

```bash
export FYP_BITBUCKET_ROOT="/vol/bitbucket/$USER/Provably-Secure-NLG"
export FYP_LARGE_OUTPUT_ROOT="$FYP_BITBUCKET_ROOT/outputs/large_experiments"
export PIP_CACHE_DIR="$FYP_BITBUCKET_ROOT/pip-cache"
export TORCH_HOME="$FYP_BITBUCKET_ROOT/torch-cache"
export HF_HOME="$FYP_BITBUCKET_ROOT/hf-cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export XDG_CACHE_HOME="$FYP_BITBUCKET_ROOT/cache"
```

The mock export, discovery, validation, schemas, and IO paths do not need
PyTorch or model caches. `FYP_LARGE_OUTPUT_ROOT` redirects relative large
experiment output paths only; local runs continue using the repository-relative
defaults when the variable is unset.

## Discovery

Discovery inspects paths without loading models:

```bash
python -m large_experiments.vpa.integration.discover_vpa \
  --adapter-dir /data/<username>/output/adapters_last3_lora \
  --test-path /data/<username>/VPA/data/test.jsonl \
  --num-shards 1 \
  --output-dir /data/<username>/output/vpa_integration_smoke \
  --cluster-username <username>
```

## Cluster-Only Real Smoke

Real inference is opt-in and intentionally tiny:

```bash
python -m large_experiments.vpa.integration.export_votes \
  --backend vpa \
  --enable-real-inference \
  --mode stability \
  --adapter-dir /data/<username>/output/adapters_last3_lora \
  --model-name allenai/OLMo-2-0425-1B-Instruct \
  --output /data/<username>/output/vpa_integration_smoke/stability_votes.jsonl \
  --num-examples 1 \
  --num-positions 1 \
  --num-shards 1 \
  --cluster-username <username>
```

This path must remain sequential: one process, one shard adapter at a time, no
training, no multiprocessing, no thread pools, no process pools, and no job
packing. Do not write large outputs to home directories. Use configured `/data`
roots for datasets and `/data/<username>/output` for generated experiment
outputs.

## Gitignore Policy

`outputs/`, `artifacts/`, optional VPA-main checkouts, generated JSONL data,
adapter directories, and large `.npz` artifacts are gitignored. `.gitkeep` files
keep empty output directories present in the source tree.
