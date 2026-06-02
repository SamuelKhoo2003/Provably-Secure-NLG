# VPA Integration Audit - Stage 7

Audit date: 2026-06-01

## 1. Files Created Or Modified

Created:

- `vpa_integration_AUDIT_stage7.md`

No integration source files were modified in Stage 7.

No changes were made to `external/VPA-main` or `toy_certificate`.

## 2. Environment Activation

No cluster virtual environment was activated because this machine does not have the expected cluster roots.

Commands run to inspect the environment:

```bash
whoami
ls -ld /data /data/samuelkhoo /data/samuelkhoo/output /vol/bitbucket/samuelkhoo
find /data/samuelkhoo -maxdepth 4 -type d -name 'adapters_last3_lora'
find /vol/bitbucket/samuelkhoo -maxdepth 3 -type f -path '*/bin/activate'
```

Observed:

```text
whoami -> samuelkhoo
/data: No such file or directory
/vol/bitbucket/samuelkhoo: No such file or directory
find: /data/samuelkhoo: No such file or directory
find: /vol/bitbucket/samuelkhoo: No such file or directory
```

Therefore no command of the form `source /vol/bitbucket/<username>/<env_name>/bin/activate` could be run in this local environment.

## 3. Discovery Command

Command run:

```bash
python3 -B -m vpa_integration.discover_vpa --adapter-dir /data/samuelkhoo/output/adapters_last3_lora --test-path /data/samuelkhoo/VPA/data/test.jsonl --num-shards 1 --output-dir /data/samuelkhoo/output/vpa_integration_smoke --cluster-username samuelkhoo --metadata-output outputs/stage7/discovery.meta.json
```

The metadata output was written inside the workspace because `/data/samuelkhoo/...` does not exist on this machine.

Discovery output summary:

```json
{
  "adapter_dir": "/data/samuelkhoo/output/adapters_last3_lora",
  "adapter_dir_exists": false,
  "num_shard_adapter_dirs_found": 0,
  "first_shard_ids": [],
  "test_path": "/data/samuelkhoo/VPA/data/test.jsonl",
  "test_path_exists": false,
  "num_test_examples": null,
  "chosen_num_shards": 1,
  "output_dir": "/data/samuelkhoo/output/vpa_integration_smoke",
  "output_dir_exists": false,
  "real_model_loading_attempted": false
}
```

Safety warning:

```text
dataset_dir: cluster path is outside preferred root /data/samuelkhoo/datasets
```

No `/homes/` path was used or written.

## 4. Real Smoke Command

The real smoke command was not attempted because discovery found zero shard adapter directories.

The command that would be run on the cluster after discovery finds at least one shard is:

```bash
python3 -B -m vpa_integration.export_votes --backend vpa --enable-real-inference --mode stability --adapter-dir /data/samuelkhoo/output/adapters_last3_lora --model-name allenai/OLMo-2-0425-1B-Instruct --output /data/samuelkhoo/output/vpa_integration_smoke/stability_votes.jsonl --num-examples 1 --num-positions 1 --num-shards 1 --cluster-username samuelkhoo
```

Result:

```text
not run; discovery found num_shard_adapter_dirs_found = 0
```

Failure reason:

```text
missing adapter directory /data/samuelkhoo/output/adapters_last3_lora in the current environment
```

No real model loading was attempted. No adapter API compatibility check was exercised against a real model. No real adapter was loaded. No next-token vote was computed. No real artifact was produced.

## 5. Real Artifact Validation

No real artifact was produced, so no real validation command was run.

## 6. Adapter Strategy And Compatibility

Configured real backend strategy:

```text
transformers_adapter_methods
```

Expected model methods:

- `load_adapter`
- `set_adapter`
- `delete_adapter`

Compatibility status:

```text
not tested in Stage 7 because no adapter directory was available and real smoke inference was not run
```

If this strategy fails on the cluster, Stage 8 should add an explicit `peft_model_from_pretrained` fallback strategy inside `vpa_integration/vpa_backend.py` only.

## 7. Mock Regression

Mock export command run:

```bash
python3 -B -m vpa_integration.export_votes --backend mock --mode stability --output outputs/stage7/mock_stability_votes.jsonl --num-examples 3 --num-positions 4 --num-shards 6
```

Output:

```text
Wrote outputs/stage7/mock_stability_votes.jsonl
Wrote outputs/stage7/mock_stability_votes.meta.json
```

Validation command run:

```bash
python3 -B -m vpa_integration.validate_votes outputs/stage7/mock_stability_votes.jsonl
```

Validation output:

```text
Validation passed for outputs/stage7/mock_stability_votes.jsonl: 12 rows
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
  "output_path": "outputs/stage7/mock_stability_votes.jsonl",
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

## 8. Safety Checks

Heavy import locations:

```bash
rg -n "^(\\s*)?(import torch|from transformers|from peft|import peft|from gurobipy|import gurobipy)" vpa_integration
```

Output:

```text
vpa_integration/vpa_backend.py:81:        import torch
vpa_integration/vpa_backend.py:82:        from transformers import AutoModelForCausalLM, AutoTokenizer
vpa_integration/vpa_backend.py:137:        import torch
```

No concurrency code:

```bash
rg -n "multiprocessing|concurrent\\.futures|ThreadPoolExecutor|ProcessPoolExecutor|joblib|ray|torch\\.distributed|accelerate|Pool\\(" vpa_integration --glob '*.py'
```

No matches.

External trees untouched:

```bash
git status --short external/VPA-main toy_certificate
```

Output was empty.

## 9. Stage 7 Outcome

Acceptance criterion status:

- One real vote row was not produced because the cluster adapter directory was unavailable in this environment.
- The exact blocker is missing `/data/samuelkhoo/output/adapters_last3_lora`.
- Adapter API compatibility was not confirmed because no real model/adapter was loaded.
- Mock backend still works and validates.
- No concurrency was introduced.
- No `/homes/` path was written.
- `external/VPA-main` was untouched.
- `toy_certificate` was untouched.

## 10. What Remains For Stage 8

- Run Stage 7 again on the actual cluster after activating the correct virtual environment.
- Ensure `/data/samuelkhoo/output/adapters_last3_lora` or the correct adapter root exists and contains at least one `shard_*` directory.
- If `transformers_adapter_methods` fails due to missing adapter methods, implement an explicit `peft_model_from_pretrained` strategy inside `vpa_integration/vpa_backend.py` only.
- Once one real row is produced, validate it with `vpa_integration.validate_votes`.
- Only after a validated real one-row artifact exists, consider scaling to tiny multi-row real exports.
