# Ada Large Experiments Storage Summary

## Files Inspected

- `large_experiments/storage.py`
- `large_experiments/vpa/README.md`
- `large_experiments/vpa/integration/config.py`
- `large_experiments/vpa/integration/discover_vpa.py`
- `large_experiments/vpa/integration/export_votes.py`
- `large_experiments/vpa/integration/generate_mock_votes.py`
- `large_experiments/vpa/integration/io.py`
- `large_experiments/vpa/integration/safety.py`
- `large_experiments/vpa/integration/validate_votes.py`
- `large_experiments/scripts/setup_ada_large_experiments.sh`
- `large_experiments/scripts/activate_ada_large_experiments.sh`
- `external/VPA-main/src/*.py`
- `.gitignore`
- `requirements.txt`

## Files Changed

- `large_experiments/scripts/activate_ada_large_experiments.sh`
- `large_experiments/vpa/integration/validate_votes.py`
- `large_experiments/vpa/README.md`
- `large_experiments/docs/ada_large_experiments.md`
- `large_experiments/docs/ada_large_experiments_storage_summary.md`

## Scripts

`large_experiments/scripts/setup_ada_large_experiments.sh` is executable and prepares:

- `/vol/bitbucket/$USER/Provably-Secure-NLG/venvs/large-experiments`
- `/vol/bitbucket/$USER/Provably-Secure-NLG/outputs/large_experiments`
- pip, Torch, HuggingFace, model, and XDG cache directories under bitbucket

It uses `python3 -m virtualenv` and does not install dependencies or PyTorch.

`large_experiments/scripts/activate_ada_large_experiments.sh` is executable but should be sourced.
It exports storage and cache environment variables without activating the venv.

## Environment Variables

- `FYP_BITBUCKET_ROOT`
- `FYP_LARGE_OUTPUT_ROOT`
- `FYP_LARGE_VENV_DIR`
- `PIP_CACHE_DIR`
- `TORCH_HOME`
- `HF_HOME`
- `TRANSFORMERS_CACHE`
- `XDG_CACHE_HOME`
- `MODEL_CACHE_DIR`

## Output Redirection

Large experiment modules use
`large_experiments.storage.resolve_large_output_path` for VPA integration output
paths and VPA vote validation artifact paths. Absolute paths are respected.
Relative paths keep local behavior when `FYP_LARGE_OUTPUT_ROOT` is unset.

When `FYP_LARGE_OUTPUT_ROOT` is set, a path beginning with
`large_experiments/` has that prefix stripped before being placed under the
configured output root. For example:

```text
large_experiments/vpa/outputs/mock_stability_votes.jsonl
```

becomes:

```text
$FYP_LARGE_OUTPUT_ROOT/vpa/outputs/mock_stability_votes.jsonl
```

The preserved external scripts under `external/VPA-main` still include upstream
hardcoded `/data/...` and `output/...` defaults. Ada users should prefer the
`large_experiments.vpa.integration` wrappers or pass explicit bitbucket paths to
external scripts that support path arguments.

## Local Behavior

Local development does not require bitbucket. If `FYP_LARGE_OUTPUT_ROOT` is not
set, large experiment outputs remain repository-relative.

## Toy Experiments

No broad `toy_experiments` changes were made.

## Manual Ada Steps

After pulling on Ada:

```bash
cd ~/Projects/Provably-Secure-NLG
bash large_experiments/scripts/setup_ada_large_experiments.sh
source /vol/bitbucket/$USER/Provably-Secure-NLG/venvs/large-experiments/bin/activate
source large_experiments/scripts/activate_ada_large_experiments.sh
```

Then install only the dependencies needed for the run into the bitbucket venv.
PyTorch is intentionally manual.
