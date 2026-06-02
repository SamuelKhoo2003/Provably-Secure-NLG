# VPA Integration Audit - Stage 4

Audit date: 2026-06-01

## 1. Files Created Or Modified

Created:

- `vpa_integration/safety.py`
- `vpa_integration/vpa_backend.py`
- `vpa_integration/discover_vpa.py`
- `vpa_integration_AUDIT_stage4.md`

Modified:

- `vpa_integration/backends.py`
- `vpa_integration/export_votes.py`
- `vpa_integration/metadata.py`
- `vpa_integration/README.md`

## 2. Generalized Safety Rules

The cluster safety rules are encoded in `vpa_integration/safety.py` without any hard-coded professor-specific username.

- `MAX_CONCURRENT_JOBS = 1`
- `CONCURRENCY_MODE = "sequential"`
- `HOME_WRITE_FORBIDDEN = True`
- configured paths under `/homes/` are errors
- local development paths are warnings, not hard failures
- preferred cluster roots are derived from `cluster_username` when supplied or inferred:
  - project/adapters: `/data/<username>/...`
  - datasets: `/data/<username>/datasets/...`
  - outputs: `/data/<username>/output/...`

## 3. Discovery Commands

Command run with local development paths:

```bash
python3 -B -m vpa_integration.discover_vpa --adapter-dir external/VPA-main/output/adapters_last3_lora --test-path external/VPA-main/data/test.jsonl --num-shards 3 --output-dir outputs/discovery/vpa_integration_smoke --cluster-username samuelkhoo --metadata-output outputs/discovery/discovery.meta.json
```

Key output:

```json
{
  "adapter_dir_exists": false,
  "num_shard_adapter_dirs_found": 0,
  "first_shard_ids": [],
  "test_path_exists": false,
  "num_test_examples": null,
  "chosen_num_shards": 3,
  "real_model_loading_attempted": false
}
```

The command wrote `outputs/discovery/discovery.meta.json`.

## 4. Mock Export Commands

Commands run:

```bash
python3 -B -m vpa_integration.export_votes --backend mock --mode stability --output outputs/export_stage4/stability_votes.jsonl --num-examples 3 --num-positions 4 --num-shards 6
python3 -B -m vpa_integration.export_votes --backend mock --mode validity --output outputs/export_stage4/validity_votes.jsonl --num-examples 3 --num-positions 4 --num-shards 6
```

Outputs:

```text
Wrote outputs/export_stage4/stability_votes.jsonl
Wrote outputs/export_stage4/stability_votes.meta.json
Wrote outputs/export_stage4/validity_votes.jsonl
Wrote outputs/export_stage4/validity_votes.meta.json
```

Validation commands:

```bash
python3 -B -m vpa_integration.validate_votes outputs/export_stage4/stability_votes.jsonl
python3 -B -m vpa_integration.validate_votes outputs/export_stage4/validity_votes.jsonl
```

Outputs:

```text
Validation passed for outputs/export_stage4/stability_votes.jsonl: 12 rows
Validation passed for outputs/export_stage4/validity_votes.jsonl: 12 rows
```

Dry-run/failure command for the real backend path:

```bash
python3 -B -m vpa_integration.export_votes --backend vpa --mode stability --output outputs/export_stage4/vpa_should_not_run.jsonl --num-examples 1 --num-positions 1 --num-shards 1
```

Output and status:

```text
Real VPA inference is not enabled in Stage 4. Use --backend mock or run discovery with vpa_integration.discover_vpa.
```

The command exited with status `2` before constructing a real backend or running inference.

## 5. Real Model Loading

No real model loading was implemented or attempted.

`vpa_integration/vpa_backend.py` defines placeholders for:

- `discover_shards()`
- `load_base_model_and_tokenizer()`
- `load_adapter_for_shard(shard_id)`
- `predict_next_token_for_shards(request, shard_ids)`

The loading and prediction methods raise `NotImplementedError` in Stage 4.

## 6. torch / transformers / PEFT Imports

No torch, transformers, PEFT, or Gurobi imports were added.

Command run:

```bash
rg -n "^(import|from) (torch|transformers|peft|gurobi|gurobipy)" vpa_integration
```

No matches.

## 7. Concurrency

No multiprocessing, process pools, thread pools, job packing, distributed execution, or concurrent GPU work was added.

Command run:

```bash
rg -n "multiprocessing|concurrent\\.futures|ThreadPoolExecutor|ProcessPoolExecutor|joblib|ray|torch\\.distributed|accelerate" vpa_integration
```

No matches.

## 8. external/VPA-main Status

`external/VPA-main` was not modified.

Command run:

```bash
git status --short external/VPA-main
```

Output was empty.

## 9. toy_certificate Status

`toy_certificate` was not modified.

Command run:

```bash
git status --short toy_certificate
```

Output was empty.

## 10. Safety Warnings

The local discovery command produced warnings because local development paths were used rather than cluster `/data` roots:

```text
dataset_dir: non-cluster/local path detected; acceptable for development but use configured /data roots for shared-server runs
adapter_dir: non-cluster/local path detected; acceptable for development but use configured /data roots for shared-server runs
output_dir: non-cluster/local path detected; acceptable for development but use configured /data roots for shared-server runs
```

No `/homes/` paths were used or written.

Stage 4 export metadata includes safety fields:

```json
{
  "safety": {
    "cluster_username": null,
    "concurrency_mode": "sequential",
    "home_write_forbidden": true,
    "max_concurrent_jobs": 1,
    "real_inference_enabled": false
  }
}
```

Example stability row from `outputs/export_stage4/stability_votes.jsonl`:

```json
{"example_id": "export_example_0000", "majority_token_id": 20000, "mode": "stability", "num_shards": 6, "position": 0, "prefix_token_ids": [101, 10000], "shard_ids": ["shard_0000", "shard_0001", "shard_0002", "shard_0003", "shard_0004", "shard_0005"], "shard_token_ids": [20000, 20000, 20001, 20002, 20000, 20000], "vote_counts": {"20000": 4, "20001": 1, "20002": 1}}
```

Example validity row from `outputs/export_stage4/validity_votes.jsonl`:

```json
{"example_id": "export_example_0000", "majority_token_id": 80000, "mode": "validity", "num_shards": 6, "position": 0, "shard_ids": ["shard_0000", "shard_0001", "shard_0002", "shard_0003", "shard_0004", "shard_0005"], "shard_token_ids": [80000, 80000, 80001, 80001, 80002, 80000], "target_id": "export_target_00", "target_prefix_token_ids": [501, 60000], "target_token_id": 80000, "vote_counts": {"80000": 3, "80001": 2, "80002": 1}}
```

## 11. Username-Specific Path Check

Command run:

```bash
rg -n "mwicker" vpa_integration vpa_integration_AUDIT_stage4.md
```

No matches.

No username-specific professor path was introduced in new Stage 4 integration code or audit.

## 12. What Remains For Stage 5

- Implement a real VPA backend only behind an explicit opt-in flag.
- Keep heavy imports isolated to `vpa_backend.py` runtime methods.
- Add tiny real-inference smoke mode with one process and sequential adapter evaluation.
- Keep discovery and validation model-free.
- Preserve shard ordering and metadata for real exports.
- Do not introduce baselines or MILPs into the exporter.
