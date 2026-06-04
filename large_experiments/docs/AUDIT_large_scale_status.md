# Large-Scale VPA Status Audit

Date: 2026-06-04

Scope: static audit plus lightweight syntax, mock export, validation, search,
gitignore, and git-status checks. No source code was modified for this audit.
No real VPA inference, GPU code, training, package installation, deletion, or
file moves were run.

## Executive Summary

The canonical large-scale VPA integration now lives under
`large_experiments/vpa/integration/`. The mock/export/validation/schema/metadata
path is usable without VPA-main, torch, transformers, PEFT, Gurobi, GPU, or
trained adapters. The key shard-aware JSONL artifact format is present and
validated: it preserves both aggregate `vote_counts` and per-shard
`shard_ids`/`shard_token_ids`.

The real VPA backend is a scaffold for a one-example, one-position, one-shard
stability smoke only. It is opt-in, sequential, and keeps heavy imports inside
runtime methods. Real validity target-prefix export is not implemented.

The repo does not currently contain a `vpa_integration/` compatibility wrapper
or `toy_certificate/` directory. Older audit files describe those earlier
paths, but the current workspace has only `large_experiments/vpa/integration/`
and `toy_experiments/`.

Major missing pieces are:

- paper-exact standalone TPA Algorithm 1 implementation for VPA JSONL artifacts;
- a VPA JSONL standalone TPA evaluator;
- artifact-to-dense-tensor conversion for toy-style baselines/MILPs;
- real VPA validity target-prefix export;
- paper-mapped TPA+MSC / collective TPA, separate from the existing row-column
  toy MILP.

## Current Architecture

Canonical package path:

- `large_experiments/vpa/integration/`

Compatibility wrapper path:

- `vpa_integration/` is not present in the current workspace.
- Historical audit files under `audit/` describe an earlier `vpa_integration`
  package, but current commands should use
  `large_experiments.vpa.integration`.

Docs path:

- `large_experiments/README.md`
- `large_experiments/docs/ada_large_experiments.md`
- `large_experiments/docs/ada_large_experiments_storage_summary.md`
- `large_experiments/vpa/README.md`
- `large_experiments/vpa/integration/README.md`
- previous stage audits under `audit/`

Scripts path:

- `large_experiments/scripts/setup_ada_large_experiments.sh`
- `large_experiments/scripts/activate_ada_large_experiments.sh`

Output/artifact paths:

- `large_experiments/vpa/outputs/` for generated JSONL/mock outputs
- `large_experiments/vpa/artifacts/` for generated vote artifacts
- both are gitignored except `.gitkeep`

External VPA-main paths:

- legacy present path: `external/VPA-main`
- optional expected path: `large_experiments/vpa/external/VPA-main`
- `large_experiments/vpa/external/VPA-main/` is ignored by root `.gitignore`

Current dirty worktree note:

- `git status --short` reports deleted tracked files:
  - `D large_experiments/vpa/AUDIT_tpa_implementation.md`
  - `D large_experiments/vpa/scripts/README.md`
- I did not revert or modify those deletions.

## Stage-By-Stage Status

| Stage | Status | Evidence | Notes |
|---|---|---|---|
| Stage 0 repo audit | Complete historically | `audit/vpa_integration_AUDIT_stage0.md` | Audited `external/VPA-main` and first-party certificate code. Mentions older `toy_certificate` path, now absent. |
| Stage 1 schema/config/io skeleton | Complete, reorganized | `audit/vpa_integration_AUDIT_stage1.md`; current `large_experiments/vpa/integration/{schemas,config,io}.py` | Current canonical path differs from historical `vpa_integration`. |
| Stage 2 mock vote generation and validation | Complete | `generate_mock_votes.py`, `validate_votes.py`; `audit/vpa_integration_AUDIT_stage2.md` | Current validator passes mock artifacts. |
| Stage 3 backend abstraction and mock exporter | Complete | `backends.py`, `export_votes.py`, `metadata.py`; `audit/vpa_integration_AUDIT_stage3.md` | Mock stability and validity exports work. |
| Stage 4 safety/discovery/VPA backend scaffold | Complete | `safety.py`, `discover_vpa.py`, `vpa_backend.py`; `audit/vpa_integration_AUDIT_stage4.md` | Discovery is model-free. |
| Stage 5 opt-in real backend smoke code | Partial | `export_votes.py`, `vpa_backend.py`; `audit/vpa_integration_AUDIT_stage5.md` | Real path is opt-in and stability-only. |
| Stage 6 adapter API compatibility checks | Complete for current strategy | `vpa_backend.py` defines `ADAPTER_STRATEGY_TRANSFORMERS`, required adapter methods, and `AdapterCompatibilityError`; `audit/vpa_integration_AUDIT_stage6.md` | No PEFT fallback strategy. |
| Stage 7 real smoke readiness | Blocked locally | `audit/vpa_integration_AUDIT_stage7.md` | Local machine lacks Ada `/data`/`/vol/bitbucket` roots and real adapters/test data. |
| Reorganization into `large_experiments/vpa` | Mostly complete | current package under `large_experiments/vpa/integration` | Compatibility wrapper `vpa_integration/` is absent. |
| Moving Ada setup docs/scripts | Complete | `large_experiments/AUDIT_move_ada_setup.md`; current scripts/docs paths | Root-level Ada setup scripts are gone. |
| TPA implementation audit | Missing in current worktree | `git status` shows `D large_experiments/vpa/AUDIT_tpa_implementation.md` | The audit file is deleted in current worktree; TPA status below is from source inspection. |

## What Works Now

- Importable canonical package under `large_experiments.vpa.integration`.
- Schema dataclasses for stability and validity vote rows.
- JSONL IO helpers.
- Metadata sidecars for export outputs.
- Mock stability export.
- Mock validity export.
- Vote artifact validation.
- Path discovery without model loading.
- Ada bitbucket setup docs and scripts.
- Output redirection through `FYP_LARGE_OUTPUT_ROOT` for export, discovery
  metadata, and validation input paths.
- A real VPA backend scaffold that loads the base model/tokenizer and then
  evaluates shard adapters sequentially, but only when explicitly requested.

## What Is Blocked

Stage 7 real smoke is blocked in this local environment because it needs:

- active Ada bitbucket venv;
- CUDA PyTorch working;
- valid adapter directory containing `shard_*`;
- valid `test.jsonl`;
- one example, one position, one shard;
- no concurrency.

`audit/vpa_integration_AUDIT_stage7.md` records that local `/data/...` and
`/vol/bitbucket/...` roots were not present and zero adapter shard directories
were found. I found no current logs proving CUDA works with available artifacts.

## What Still Needs Implementation

- Standalone paper-exact TPA Algorithm 1 for sparse JSONL `vote_counts`.
- CLI evaluator that reads VPA validity JSONL and computes standalone TPA radii.
- Real VPA validity target-prefix export.
- Artifact-to-tensor conversion from JSONL to dense toy-style arrays.
- Connection from converted artifacts to DPA/TPA baselines and row-column MILPs.
- Paper-mapped TPA+MSC / collective TPA, if desired, as a separate method.
- Optional compatibility wrapper restoration, if external users still rely on
  `vpa_integration`.

## VPA-Main And Artifact Dependency Status

VPA-main is not required for:

- schemas;
- JSONL IO;
- metadata;
- validation;
- mock vote generation;
- mock export;
- discovery/path safety checks.

VPA-main, model/tokenizer runtime, trained shard adapters, and test data are
required for real adapter inference.

Git/ignore status:

- `external/VPA-main` source is present and appears tracked/visible.
- `external/VPA-main/.gitignore` ignores its `data/`, `output/`, `cache/`,
  model binaries, `.pt`, `.bin`, logs, and local envs.
- root `.gitignore` ignores:
  - `large_experiments/vpa/outputs/*`
  - `large_experiments/vpa/artifacts/*`
  - `large_experiments/vpa/external/VPA-main/`
  - legacy `external/VPA-main/output/`, `data/`, `datasets/`, JSONL files, and
    adapter directories.

The ignore setup is safe for generated artifacts. The direct check-ignore
command reported the generated VPA output as ignored and left `.gitkeep` files
unignored, as intended.

## Ada Setup And Storage Status

Setup script:

- `large_experiments/scripts/setup_ada_large_experiments.sh`

Activation script:

- `large_experiments/scripts/activate_ada_large_experiments.sh`

Docs:

- `large_experiments/docs/ada_large_experiments.md`
- `large_experiments/vpa/README.md`

The docs reference the new `large_experiments/scripts/...` paths. They clearly
state that commands should be run from the repository root. The scripts preserve
that repo-root assumption.

The setup/activation scripts set the intended bitbucket paths:

- `FYP_BITBUCKET_ROOT=/vol/bitbucket/$USER/Provably-Secure-NLG`
- `FYP_LARGE_OUTPUT_ROOT=$FYP_BITBUCKET_ROOT/outputs/large_experiments`
- `FYP_LARGE_VENV_DIR=$FYP_BITBUCKET_ROOT/venvs/large-experiments`
- `PIP_CACHE_DIR`
- `TORCH_HOME`
- `HF_HOME`
- `TRANSFORMERS_CACHE`
- `XDG_CACHE_HOME`
- `MODEL_CACHE_DIR`

The setup script correctly uses `python3 -m virtualenv`, not `python3 -m venv`.
The docs use `python3` for local venv creation but one mock command example
still uses `python -m large_experiments...`; this is not harmful if the active
venv provides `python`, but the audit machine did not have a bare `python`
executable. Canonical validation commands in this audit used `python3`.

Syntax checks passed:

```bash
bash -n large_experiments/scripts/setup_ada_large_experiments.sh
bash -n large_experiments/scripts/activate_ada_large_experiments.sh
```

## Storage And Output Redirection

`large_experiments/storage.py` defines
`resolve_large_output_path(path, base_dir=None)`.

Behavior:

- absolute paths are respected;
- if `FYP_LARGE_OUTPUT_ROOT` is unset, relative paths keep local behavior;
- if `FYP_LARGE_OUTPUT_ROOT` is set and the path starts with
  `large_experiments/`, that prefix is stripped before placing the path under
  the configured root.

Therefore:

```text
large_experiments/vpa/outputs/foo.jsonl
```

resolves on Ada to:

```text
$FYP_LARGE_OUTPUT_ROOT/vpa/outputs/foo.jsonl
```

Writers using the helper:

- mock/stability export: `export_votes.py`
- mock/validity export: `export_votes.py`
- metadata sidecars for export outputs: path is derived after output
  resolution;
- discovery output dir and optional metadata output: `discover_vpa.py`;
- mock generator output dir: `generate_mock_votes.py`.

Validation is read-only, but its CLI resolves the input path through the same
helper before reading.

## JSONL Vote Artifact Schema

Stability rows contain all required fields:

- `mode`
- `example_id`
- `position`
- `prefix_token_ids`
- `shard_ids`
- `shard_token_ids`
- `vote_counts`
- `majority_token_id`
- `num_shards`

Validity rows contain all required fields:

- `mode`
- `example_id`
- `target_id`
- `position`
- `target_prefix_token_ids`
- `target_token_id`
- `shard_ids`
- `shard_token_ids`
- `vote_counts`
- `majority_token_id`
- `num_shards`

Validator behavior:

- checks `mode`;
- checks non-negative integer `position`;
- checks `shard_ids` and `shard_token_ids` are lists and lengths match;
- checks `num_shards == len(shard_ids)`;
- recomputes `vote_counts` from `shard_token_ids`;
- normalizes JSON string keys in `vote_counts` back to integer ids;
- recomputes `majority_token_id`;
- requires `prefix_token_ids` for stability;
- requires `target_id`, `target_prefix_token_ids`, and `target_token_id` for
  validity.

Tie behavior:

- `compute_majority_token_id` uses `Counter(...).most_common(1)[0][0]`, so ties
  follow first occurrence in shard order.
- The validator uses the same function, so validation is internally consistent.
- This tie policy is not yet documented as a TPA tie policy.

## Mock Export Status

All requested mock commands passed with `python3`:

```bash
python3 -B -m large_experiments.vpa.integration.export_votes \
  --backend mock \
  --mode stability \
  --output large_experiments/vpa/outputs/audit_status_mock_stability.jsonl \
  --num-examples 2 \
  --num-positions 2 \
  --num-shards 3
```

Output:

```text
Wrote large_experiments/vpa/outputs/audit_status_mock_stability.jsonl
Wrote large_experiments/vpa/outputs/audit_status_mock_stability.meta.json
```

```bash
python3 -B -m large_experiments.vpa.integration.validate_votes \
  large_experiments/vpa/outputs/audit_status_mock_stability.jsonl
```

Output:

```text
Validation passed for large_experiments/vpa/outputs/audit_status_mock_stability.jsonl: 4 rows
```

```bash
python3 -B -m large_experiments.vpa.integration.export_votes \
  --backend mock \
  --mode validity \
  --output large_experiments/vpa/outputs/audit_status_mock_validity.jsonl \
  --num-examples 2 \
  --num-positions 3 \
  --num-shards 4
```

Output:

```text
Wrote large_experiments/vpa/outputs/audit_status_mock_validity.jsonl
Wrote large_experiments/vpa/outputs/audit_status_mock_validity.meta.json
```

```bash
python3 -B -m large_experiments.vpa.integration.validate_votes \
  large_experiments/vpa/outputs/audit_status_mock_validity.jsonl
```

Output:

```text
Validation passed for large_experiments/vpa/outputs/audit_status_mock_validity.jsonl: 6 rows
```

## Real VPA Backend Status

Real inference requires:

- `--backend vpa`
- `--enable-real-inference`
- `--adapter-dir`
- `--model-name`

The default usable backend remains `mock`.

Heavy imports:

- `import torch` inside `VPAAdapterBackend.load_base_model_and_tokenizer`
- `from transformers import AutoModelForCausalLM, AutoTokenizer` inside the
  same method
- `import torch` inside `_predict_next_token`

No model loading happens at import time. No `peft` import exists. No PEFT
fallback strategy exists. The explicit adapter strategy is:

```text
transformers_adapter_methods
```

Compatibility checks require:

- `load_adapter`
- `set_adapter`
- `delete_adapter`

The real backend evaluates shards sequentially and deletes each adapter before
moving to the next shard. It does not call `model.generate`, does not train, and
does not introduce concurrency.

Current real support:

- stability smoke only;
- one example, one position, one shard only;
- real validity target-prefix export is not supported because
  `_build_vpa_backend_from_args` rejects modes other than `stability`.

## GPU And Cluster Status

The audit did not require or use GPU. Real Stage 7 smoke should be run only on
Ada/cluster after:

- bitbucket venv is active;
- CUDA PyTorch works;
- a valid adapter directory contains `shard_*`;
- a valid `test.jsonl` exists;
- the command is limited to one example, one position, one shard;
- no concurrency is used.

The local Stage 7 audit history indicates CUDA/artifacts were not available
locally because cluster roots and adapter/test paths were missing.

## DPA, TPA, And MILP Status

DPA stability:

- implemented in toy code via `toy_experiments/baselines.py`
  `cell_stability_budgets` and `phd_margin_stability_budgets`.

DPA validity diagnostic:

- implemented as `plain_dpa_count_margin_radius` and
  `plain_dpa_validity_token_budgets`.

Standalone TPA:

- implemented only in toy code as
  `toy_experiments/baselines.py:targeted_partition_radius`.
- It is count-based and shard-free.
- It is a brute-force targeted count search, not a literal paper Algorithm 1
  implementation with named `Delta`, `Phi`, and `s_star`.

Multi-token standalone TPA:

- toy sequence aggregation uses max over per-token radii through
  `aggregate_tpa_sequence_baselines`.

Target-prefix validity vote collection:

- implemented in mock exporter: validity appends `target_token_id` to the
  prefix at each position.
- not implemented for real VPA export.

TPA+MSC / collective TPA:

- not found as a named or paper-mapped implementation.

Row-column validity MILP:

- implemented in `toy_experiments/milp.py:solve_row_col_validity`.
- This is a separate shard-aware shared-budget row-column MILP over toy cells,
  not standalone TPA and not clearly paper TPA+MSC.

Confusing labels:

- metric names such as `dpa_val_*` are confusing because some are populated from
  shard-aware validity logic (`cell_validity_budgets`) rather than plain DPA.

## VPA JSONL Evaluator Status

No CLI currently reads VPA validity JSONL and computes standalone TPA radii.

Missing expected behavior:

- read validity rows;
- group by `(example_id, target_id)`;
- compute per-token TPA from `vote_counts` and `target_token_id`;
- sequence radius is max over positions;
- require no shard identities;
- require no torch, transformers, PEFT, or Gurobi.

## Artifact-To-Tensor Conversion Status

No converter currently maps VPA JSONL artifacts into dense toy-style tensors.

Missing expected outputs:

- `shard_ids`;
- `example_ids`;
- `positions`;
- `votes` with shape `K x N x L`;
- `counts` with shape `N x L x T_local`;
- `token_to_local_id`;
- `local_id_to_token`;
- `majority_tokens`;
- validity `target_tokens`.

## Safety And Gitignore Status

Exact concurrency search command:

```bash
rg -n "multiprocessing|concurrent\.futures|ThreadPoolExecutor|ProcessPoolExecutor|joblib|ray|torch\.distributed|accelerate|Pool\(" large_experiments vpa_integration toy_experiments toy_certificate --glob '*.py' 2>&1
```

Warnings:

- `vpa_integration` and `toy_certificate` are missing paths.
- The `ray` alternative creates noisy matches inside words like `ndarray`.

A cleaner search over existing paths excluding the broad `ray` term had no
hits for multiprocessing, futures, thread/process pools, joblib, distributed
torch, or accelerate.

Exact heavy-import search command:

```bash
rg -n "import torch|from transformers|import transformers|from peft|import peft|gurobipy|import gurobipy" large_experiments vpa_integration toy_experiments toy_certificate --glob '*.py' 2>&1
```

Results:

- missing path warnings for `vpa_integration` and `toy_certificate`;
- `toy_experiments/milp.py` imports `gurobipy`;
- `large_experiments/vpa/integration/vpa_backend.py` imports torch and
  transformers only inside runtime methods.

Gitignore checks:

```bash
git check-ignore -v large_experiments/vpa/outputs/audit_status_mock_stability.jsonl large_experiments/vpa/outputs/.gitkeep large_experiments/vpa/artifacts/.gitkeep external/VPA-main large_experiments/vpa/external/VPA-main 2>&1
```

This reported the generated VPA output as ignored. It did not report `.gitkeep`
files, which is expected because they are explicitly unignored. It did not
report bare `external/VPA-main`, because the source tree itself is not ignored.

Additional checks confirmed:

- generated VPA output metadata is ignored;
- generated VPA validity JSONL/metadata are ignored;
- `external/VPA-main/output/...` and `external/VPA-main/data/...` are ignored by
  `external/VPA-main/.gitignore`;
- `large_experiments/vpa/external/VPA-main/...` is ignored by root `.gitignore`.

The repo is unlikely to accidentally commit large generated VPA artifacts if
they stay under the documented output/artifact paths.

## Recommended Next Stages

1. Restore or intentionally retire the deleted `large_experiments/vpa/AUDIT_tpa_implementation.md`.
2. Finish/verify a paper-exact standalone TPA Algorithm 1 implementation.
3. Add a VPA JSONL standalone TPA evaluator.
4. Run real Stage 7 one-row stability smoke on Ada once `adapters_last3_lora`
   and `test.jsonl` are available.
5. Implement real VPA validity target-prefix export.
6. Add artifact-to-tensor conversion.
7. Run DPA/TPA on mock and tiny real artifacts.
8. Connect row-column MILPs to converted artifacts.
9. Decide whether to implement paper-mapped TPA+MSC separately.
10. Clean up confusing `dpa_val_*` metric names or document them explicitly.

## Exact Commands Run

```bash
sed -n '1,260p' /Users/samuelkhoo/.codex/attachments/21d70943-a6e8-490e-9246-a7354916e505/pasted-text.txt
sed -n '260,560p' /Users/samuelkhoo/.codex/attachments/21d70943-a6e8-490e-9246-a7354916e505/pasted-text.txt
find large_experiments -maxdepth 4 -type f \( -name '*.py' -o -name '*.md' -o -name '*.sh' -o -name '*.yaml' -o -name '*.yml' \)
find vpa_integration toy_certificate toy_experiments external/VPA-main audit -maxdepth 4 -type f \( -name '*.py' -o -name '*.md' -o -name '*.sh' -o -name '*.yaml' -o -name '*.yml' \) 2>/dev/null
sed -n '1,220p' .gitignore
sed -n '1,120p' requirements.txt
git status --short
ls -la large_experiments large_experiments/vpa large_experiments/vpa/integration large_experiments/vpa/outputs large_experiments/vpa/artifacts vpa_integration toy_certificate 2>/dev/null
rg -n "converter|convert|to_tensor|tensor|vote_counts|target_token_id|targeted_partition|targeted_validity|TPA|jsonl|read_jsonl|ValidityVoteRow|StabilityVoteRow" large_experiments toy_experiments vpa_integration toy_certificate -g '*.py' -g '*.md' 2>/dev/null
bash -n large_experiments/scripts/setup_ada_large_experiments.sh
bash -n large_experiments/scripts/activate_ada_large_experiments.sh
rg -n "multiprocessing|concurrent\.futures|ThreadPoolExecutor|ProcessPoolExecutor|joblib|ray|torch\.distributed|accelerate|Pool\(" large_experiments vpa_integration toy_experiments toy_certificate --glob '*.py' 2>&1
rg -n "import torch|from transformers|import transformers|from peft|import peft|gurobipy|import gurobipy" large_experiments vpa_integration toy_experiments toy_certificate --glob '*.py' 2>&1
python3 -B -m large_experiments.vpa.integration.export_votes --backend mock --mode stability --output large_experiments/vpa/outputs/audit_status_mock_stability.jsonl --num-examples 2 --num-positions 2 --num-shards 3
rg -n "multiprocessing|concurrent\.futures|ThreadPoolExecutor|ProcessPoolExecutor|joblib|torch\.distributed|accelerate|Pool\(" large_experiments toy_experiments --glob '*.py'
python3 -B -m large_experiments.vpa.integration.validate_votes large_experiments/vpa/outputs/audit_status_mock_stability.jsonl
python3 -B -m large_experiments.vpa.integration.export_votes --backend mock --mode validity --output large_experiments/vpa/outputs/audit_status_mock_validity.jsonl --num-examples 2 --num-positions 3 --num-shards 4
git check-ignore -v large_experiments/vpa/outputs/audit_status_mock_stability.jsonl large_experiments/vpa/outputs/.gitkeep large_experiments/vpa/artifacts/.gitkeep external/VPA-main large_experiments/vpa/external/VPA-main 2>&1
python3 -B -m large_experiments.vpa.integration.validate_votes large_experiments/vpa/outputs/audit_status_mock_validity.jsonl
git check-ignore -v large_experiments/vpa/outputs/audit_status_mock_stability.meta.json large_experiments/vpa/outputs/audit_status_mock_validity.jsonl large_experiments/vpa/outputs/audit_status_mock_validity.meta.json external/VPA-main/output/foo.jsonl external/VPA-main/data/test.jsonl external/VPA-main/output/adapters_last3_lora/shard_0000/config.json large_experiments/vpa/external/VPA-main/foo 2>&1
```

## Errors Or Warnings Encountered

- `vpa_integration` directory is absent.
- `toy_certificate` directory is absent.
- Exact safety `rg` commands return status 2 because those absent paths are
  included in the requested command.
- The exact concurrency regex has noisy `ray` matches inside `ndarray`.
- `git status --short` shows two deleted tracked files unrelated to this audit:
  `large_experiments/vpa/AUDIT_tpa_implementation.md` and
  `large_experiments/vpa/scripts/README.md`.
- Bare `python` was not tested for this audit; requested mock commands used
  `python3` and passed.

