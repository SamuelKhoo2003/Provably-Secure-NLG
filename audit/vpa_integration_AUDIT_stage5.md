# VPA Integration Audit - Stage 5

Audit date: 2026-06-01

## 1. Files Created Or Modified

Created:

- `vpa_integration_AUDIT_stage5.md`

Modified:

- `vpa_integration/vpa_backend.py`
- `vpa_integration/export_votes.py`
- `vpa_integration/backends.py`
- `vpa_integration/metadata.py`
- `vpa_integration/safety.py`
- `vpa_integration/README.md`

`vpa_integration/config.py` was not changed.

## 2. Real Inference Code

Real inference code was implemented behind explicit opt-in only.

`VPAAdapterBackend` now supports:

- `discover_shards()`
- `load_base_model_and_tokenizer()`
- `load_adapter_for_shard(shard_id)`
- `predict_next_token_for_shards(request, shard_ids)`

The backend evaluates shards sequentially. For each shard it loads one adapter, computes one next-token argmax from the prefix, deletes that adapter, and moves to the next shard.

The exporter requires:

- `--backend vpa`
- `--enable-real-inference`
- `--adapter-dir`
- `--model-name`

Stage 5 also restricts real VPA smoke export to:

- `--num-examples 1`
- `--num-positions 1`
- `--num-shards 1`
- `--mode stability`

## 3. Heavy Imports

Heavy imports are isolated to `vpa_integration/vpa_backend.py`.

Exact locations:

- `vpa_integration/vpa_backend.py`, inside `load_base_model_and_tokenizer()`:
  - `import torch`
  - `from transformers import AutoModelForCausalLM, AutoTokenizer`
- `vpa_integration/vpa_backend.py`, inside `_predict_next_token()`:
  - `import torch`

PEFT is not imported directly. The backend expects the loaded model/runtime to provide adapter methods such as `load_adapter`, `set_adapter`, and `delete_adapter`.

## 4. Lightweight Imports

Command run:

```bash
python3 -B -c "import vpa_integration; import vpa_integration.export_votes; import vpa_integration.discover_vpa; print('imports ok')"
```

Output:

```text
imports ok
```

This import check did not load models.

## 5. Discovery Remains Model-Free

`vpa_integration/discover_vpa.py` was not changed and still has no torch, transformers, PEFT, or model-loading imports.

## 6. Concurrency

No concurrency was added.

The implementation does not use multiprocessing, thread pools, process pools, job packing, distributed execution, or adapter batching.

## 7. external/VPA-main Status

`external/VPA-main` was not modified.

## 8. toy_certificate Status

`toy_certificate` was not modified.

## 9. Real Smoke Command

Command attempted:

```bash
python3 -B -m vpa_integration.export_votes --backend vpa --enable-real-inference --mode stability --output outputs/export_stage5/real_smoke/stability_votes.jsonl --num-examples 1 --num-positions 1 --num-shards 1 --adapter-dir external/VPA-main/output/adapters_last3_lora --model-name allenai/OLMo-2-0425-1B-Instruct --cluster-username samuelkhoo
```

Result:

```text
Adapter directory does not exist: external/VPA-main/output/adapters_last3_lora
```

Exit status was `2`. Real smoke inference was not executed because the adapter directory is not present in this workspace. No model loading was attempted.

Control command without opt-in:

```bash
python3 -B -m vpa_integration.export_votes --backend vpa --mode stability --output outputs/export_stage5/should_not_run.jsonl --num-examples 1 --num-positions 1 --num-shards 1 --adapter-dir external/VPA-main/output/adapters_last3_lora --model-name allenai/OLMo-2-0425-1B-Instruct
```

Output:

```text
Real VPA inference requires --enable-real-inference. Use --backend mock or run discovery first.
```

Exit status was `2`.

## 10. Real Artifact Validation

No real artifact was produced, so no real artifact validation command was run.

## 11. Mock Regression

Commands run:

```bash
python3 -B -m vpa_integration.export_votes --backend mock --mode stability --output outputs/export_stage5/mock/stability_votes.jsonl --num-examples 3 --num-positions 4 --num-shards 6
python3 -B -m vpa_integration.export_votes --backend mock --mode validity --output outputs/export_stage5/mock/validity_votes.jsonl --num-examples 3 --num-positions 4 --num-shards 6
python3 -B -m vpa_integration.validate_votes outputs/export_stage5/mock/stability_votes.jsonl
python3 -B -m vpa_integration.validate_votes outputs/export_stage5/mock/validity_votes.jsonl
```

Validation outputs:

```text
Validation passed for outputs/export_stage5/mock/stability_votes.jsonl: 12 rows
Validation passed for outputs/export_stage5/mock/validity_votes.jsonl: 12 rows
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
  "notes": "Mock stability export; no real model inference. Majority tokens are committed to the next clean prefix.",
  "num_examples": 3,
  "num_positions": 4,
  "num_shards": 6,
  "output_path": "outputs/export_stage5/mock/stability_votes.jsonl",
  "safety": {
    "cluster_username": null,
    "concurrency_mode": "sequential",
    "home_write_forbidden": true,
    "max_concurrent_jobs": 1,
    "real_inference_enabled": false
  },
  "schema_version": "vpa-token-votes/v1"
}
```

## 12. Static Checks

Command run:

```bash
rg -n "torch|transformers|peft|Peft|AutoModel|AutoTokenizer" vpa_integration
```

The only code matches were in `vpa_integration/vpa_backend.py`, plus documentation references.

Command run:

```bash
rg -n "multiprocessing|concurrent\\.futures|ThreadPoolExecutor|ProcessPoolExecutor|joblib|ray|torch\\.distributed|accelerate|Pool\\(" vpa_integration
```

This produced one documentation-only README match mentioning that multiprocessing is not used. The code-only check was:

```bash
rg -n "multiprocessing|concurrent\\.futures|ThreadPoolExecutor|ProcessPoolExecutor|joblib|ray|torch\\.distributed|accelerate|Pool\\(" vpa_integration --glob '*.py'
```

No matches in Python code.

Professor-specific username string check:

```bash
rg -n "<professor-specific username>" vpa_integration
```

No professor-specific username/path was introduced in code.

## 13. What Remains For Stage 6

- Run a real smoke export on a machine with the model, GPU/runtime, and a valid adapter directory.
- Confirm adapter API compatibility in the actual environment.
- Validate the real artifact.
- Add real prompt/tokenization support if synthetic prefixes are insufficient.
- Keep real exports sequential and limited until smoke behavior is proven.
- Do not add baselines or MILPs until real vote artifacts are validated.
