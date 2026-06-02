# VPA Integration Audit - Stage 6

Audit date: 2026-06-01

## 1. Files Created Or Modified

Created:

- `vpa_integration_AUDIT_stage6.md`

Modified:

- `vpa_integration/vpa_backend.py`
- `vpa_integration/export_votes.py`
- `vpa_integration/metadata.py`
- `vpa_integration/README.md`

No changes were made to `external/VPA-main` or `toy_certificate`.

## 2. Adapter API Checks

Adapter API compatibility checks were added.

`vpa_integration/vpa_backend.py` now defines:

- `ADAPTER_STRATEGY_TRANSFORMERS = "transformers_adapter_methods"`
- `REQUIRED_ADAPTER_METHODS = ("load_adapter", "set_adapter", "delete_adapter")`
- `AdapterCompatibilityError`

Before loading an adapter, `VPAAdapterBackend` checks:

- `hasattr(model, "load_adapter")`
- `hasattr(model, "set_adapter")`
- `hasattr(model, "delete_adapter")`

If any method is missing, it raises a clear compatibility error explaining that the runtime does not expose the expected adapter API.

## 3. Adapter Loading Strategy

The implemented strategy is:

```text
transformers_adapter_methods
```

This means the backend expects the loaded model object to expose `load_adapter`, `set_adapter`, and `delete_adapter`.

PEFT is not imported directly in this stage. No `peft_model_from_pretrained` strategy was added.

## 4. Heavy Imports

Heavy imports remain isolated to `vpa_integration/vpa_backend.py`.

Exact locations:

- `vpa_integration/vpa_backend.py:81`
  - `import torch`
- `vpa_integration/vpa_backend.py:82`
  - `from transformers import AutoModelForCausalLM, AutoTokenizer`
- `vpa_integration/vpa_backend.py:137`
  - `import torch`

Static check command:

```bash
rg -n "^(\\s*)?(import torch|from transformers|from peft|import peft|from gurobipy|import gurobipy)" vpa_integration
```

Output:

```text
vpa_integration/vpa_backend.py:81:        import torch
vpa_integration/vpa_backend.py:82:        from transformers import AutoModelForCausalLM, AutoTokenizer
vpa_integration/vpa_backend.py:137:        import torch
```

## 5. Lightweight Imports

Command run:

```bash
python3 -B -c "import vpa_integration; import vpa_integration.discover_vpa; import vpa_integration.vpa_backend; import vpa_integration.export_votes; print('stage6 imports ok')"
```

Output:

```text
stage6 imports ok
```

This confirms importing `vpa_integration` and the CLI modules does not load models.

## 6. Discovery Remains Model-Free

`vpa_integration/discover_vpa.py` remains model-free. It imports only stdlib helpers plus metadata and safety utilities.

Discovery command run:

```bash
python3 -B -m vpa_integration.discover_vpa --adapter-dir external/VPA-main/output/adapters_last3_lora --test-path external/VPA-main/data/test.jsonl --num-shards 1 --output-dir outputs/discovery_stage6/vpa_integration_smoke --cluster-username samuelkhoo --metadata-output outputs/discovery_stage6/discovery.meta.json
```

Key output:

```json
{
  "adapter_dir_exists": false,
  "num_shard_adapter_dirs_found": 0,
  "first_shard_ids": [],
  "test_path_exists": false,
  "num_test_examples": null,
  "chosen_num_shards": 1,
  "real_model_loading_attempted": false
}
```

Safety warnings were local-development warnings only:

```text
dataset_dir: non-cluster/local path detected; acceptable for development but use configured /data roots for shared-server runs
adapter_dir: non-cluster/local path detected; acceptable for development but use configured /data roots for shared-server runs
output_dir: non-cluster/local path detected; acceptable for development but use configured /data roots for shared-server runs
```

## 7. Concurrency

No concurrency was added.

Command run:

```bash
rg -n "multiprocessing|concurrent\\.futures|ThreadPoolExecutor|ProcessPoolExecutor|joblib|ray|torch\\.distributed|accelerate|Pool\\(" vpa_integration --glob '*.py'
```

No matches.

The backend still evaluates shard ids sequentially and Stage 6 real smoke export is restricted to one shard.

## 8. external/VPA-main Status

Command run:

```bash
git status --short external/VPA-main
```

Output was empty. `external/VPA-main` was untouched.

## 9. toy_certificate Status

Command run:

```bash
git status --short toy_certificate
```

Output was empty. `toy_certificate` was untouched.

## 10. Real Smoke Command

No-opt-in guard command:

```bash
python3 -B -m vpa_integration.export_votes --backend vpa --mode stability --adapter-dir external/VPA-main/output/adapters_last3_lora --model-name allenai/OLMo-2-0425-1B-Instruct --output outputs/export_stage6/no_opt_in/should_not_run.jsonl --num-examples 1 --num-positions 1 --num-shards 1 --cluster-username samuelkhoo
```

Output:

```text
Real VPA inference requires --enable-real-inference. Use --backend mock or run discovery first.
```

Exit status was `2`.

Real smoke command attempted:

```bash
python3 -B -m vpa_integration.export_votes --backend vpa --enable-real-inference --mode stability --adapter-dir external/VPA-main/output/adapters_last3_lora --model-name allenai/OLMo-2-0425-1B-Instruct --output outputs/export_stage6/real_smoke/stability_votes.jsonl --num-examples 1 --num-positions 1 --num-shards 1 --cluster-username samuelkhoo
```

Output:

```text
Adapter directory does not exist: external/VPA-main/output/adapters_last3_lora
```

Exit status was `2`.

The real smoke command failed before model loading because the adapter directory is missing in this workspace. No real adapter was loaded, no next-token inference was run, and no real artifact was produced.

## 11. Real Artifact Validation

No real artifact was produced, so there was no real validation command to run.

Confirmed no real output file exists:

```bash
test -e outputs/export_stage6/real_smoke/stability_votes.jsonl
```

Exit status was non-zero.

## 12. Mock Regression

Mock export commands run:

```bash
python3 -B -m vpa_integration.export_votes --backend mock --mode stability --output outputs/export_stage6/mock/stability_votes.jsonl --num-examples 3 --num-positions 4 --num-shards 6
python3 -B -m vpa_integration.export_votes --backend mock --mode validity --output outputs/export_stage6/mock/validity_votes.jsonl --num-examples 3 --num-positions 4 --num-shards 6
```

Validation commands run:

```bash
python3 -B -m vpa_integration.validate_votes outputs/export_stage6/mock/stability_votes.jsonl
python3 -B -m vpa_integration.validate_votes outputs/export_stage6/mock/validity_votes.jsonl
```

Validation output:

```text
Validation passed for outputs/export_stage6/mock/stability_votes.jsonl: 12 rows
Validation passed for outputs/export_stage6/mock/validity_votes.jsonl: 12 rows
```

Example mock row:

```json
{"example_id": "export_example_0000", "majority_token_id": 20000, "mode": "stability", "num_shards": 6, "position": 0, "prefix_token_ids": [101, 10000], "shard_ids": ["shard_0000", "shard_0001", "shard_0002", "shard_0003", "shard_0004", "shard_0005"], "shard_token_ids": [20000, 20000, 20001, 20002, 20000, 20000], "vote_counts": {"20000": 4, "20001": 1, "20002": 1}}
```

Example mock metadata:

```json
{
  "backend": "mock",
  "mode": "stability",
  "num_examples": 3,
  "num_positions": 4,
  "num_shards": 6,
  "output_path": "outputs/export_stage6/mock/stability_votes.jsonl",
  "safety": {
    "cluster_username": null,
    "concurrency_mode": "sequential",
    "home_write_forbidden": true,
    "max_concurrent_jobs": 1,
    "real_inference_enabled": false
  },
  "schema_version": "vpa-token-votes/v1",
  "selected_shard_ids": ["shard_0000", "shard_0001", "shard_0002", "shard_0003", "shard_0004", "shard_0005"]
}
```

## 13. Professor-Specific Path Check

Command run:

```bash
rg -n "<professor-specific username>" vpa_integration
```

No professor-specific username/path was introduced in integration code.

## 14. What Remains For Stage 7

- Run the exact one-shard real smoke command on the cluster where adapters, model cache, GPU/runtime, and adapter API are available.
- If `transformers_adapter_methods` is unsupported in that runtime, add an explicit `peft_model_from_pretrained` strategy inside `vpa_backend.py` only.
- Validate a produced real one-row artifact.
- Preserve strict sequential execution and one-adapter-at-a-time loading.
- Only after a real artifact validates, add artifact-to-tensor conversion for downstream baselines and MILPs.
