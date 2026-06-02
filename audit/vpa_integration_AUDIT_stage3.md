# VPA Integration Audit - Stage 3

Audit date: 2026-06-01

## 1. Files Created Or Modified

Created:

- `vpa_integration/backends.py`
- `vpa_integration/export_votes.py`
- `vpa_integration/metadata.py`
- `vpa_integration_AUDIT_stage3.md`

Modified:

- `vpa_integration/README.md`

## 2. Backend Abstraction Added

`vpa_integration/backends.py` defines:

- `VoteRequest`
  - Dataclass carrying `mode`, `example_id`, `position`, `prefix_token_ids`, and optional target context.
- `TokenVoteBackend`
  - Protocol with `predict_next_token_for_shards(request, shard_ids) -> list[int]`.
- `MockTokenVoteBackend`
  - Deterministic backend that returns one integer token id per shard.
  - Uses only request context and shard ids.
  - Does not import or call VPA-main, torch, transformers, PEFT, or Gurobi.
- `make_backend(...)`
  - Factory currently supporting only `mock`.

## 3. Export Commands

Commands run:

```bash
python3 -B -m vpa_integration.export_votes --backend mock --mode stability --output outputs/export_mock/stability_votes.jsonl --num-examples 3 --num-positions 4 --num-shards 6
python3 -B -m vpa_integration.export_votes --backend mock --mode validity --output outputs/export_mock/validity_votes.jsonl --num-examples 3 --num-positions 4 --num-shards 6
```

Outputs:

```text
Wrote outputs/export_mock/stability_votes.jsonl
Wrote outputs/export_mock/stability_votes.meta.json
Wrote outputs/export_mock/validity_votes.jsonl
Wrote outputs/export_mock/validity_votes.meta.json
```

## 4. Validation Commands

Commands run:

```bash
python3 -B -m vpa_integration.validate_votes outputs/export_mock/stability_votes.jsonl
python3 -B -m vpa_integration.validate_votes outputs/export_mock/validity_votes.jsonl
```

Outputs:

```text
Validation passed for outputs/export_mock/stability_votes.jsonl: 12 rows
Validation passed for outputs/export_mock/validity_votes.jsonl: 12 rows
```

## 5. Validation Result

Passed.

Both exported mock artifacts validate with the existing Stage 2 validator.

## 6. Example Rows

First three rows from `outputs/export_mock/stability_votes.jsonl`:

```json
{"example_id": "export_example_0000", "majority_token_id": 20000, "mode": "stability", "num_shards": 6, "position": 0, "prefix_token_ids": [101, 10000], "shard_ids": ["shard_0000", "shard_0001", "shard_0002", "shard_0003", "shard_0004", "shard_0005"], "shard_token_ids": [20000, 20000, 20001, 20002, 20000, 20000], "vote_counts": {"20000": 4, "20001": 1, "20002": 1}}
{"example_id": "export_example_0000", "majority_token_id": 20001, "mode": "stability", "num_shards": 6, "position": 1, "prefix_token_ids": [101, 10000, 20000], "shard_ids": ["shard_0000", "shard_0001", "shard_0002", "shard_0003", "shard_0004", "shard_0005"], "shard_token_ids": [20001, 20001, 20002, 20003, 20001, 20001], "vote_counts": {"20001": 4, "20002": 1, "20003": 1}}
{"example_id": "export_example_0000", "majority_token_id": 20002, "mode": "stability", "num_shards": 6, "position": 2, "prefix_token_ids": [101, 10000, 20000, 20001], "shard_ids": ["shard_0000", "shard_0001", "shard_0002", "shard_0003", "shard_0004", "shard_0005"], "shard_token_ids": [20002, 20002, 20003, 20004, 20002, 20002], "vote_counts": {"20002": 4, "20003": 1, "20004": 1}}
```

The stability prefix at position 1 includes the position 0 majority token `20000`; the prefix at position 2 includes `20000, 20001`.

First three rows from `outputs/export_mock/validity_votes.jsonl`:

```json
{"example_id": "export_example_0000", "majority_token_id": 80000, "mode": "validity", "num_shards": 6, "position": 0, "shard_ids": ["shard_0000", "shard_0001", "shard_0002", "shard_0003", "shard_0004", "shard_0005"], "shard_token_ids": [80000, 80000, 80001, 80001, 80002, 80000], "target_id": "export_target_00", "target_prefix_token_ids": [501, 60000], "target_token_id": 80000, "vote_counts": {"80000": 3, "80001": 2, "80002": 1}}
{"example_id": "export_example_0000", "majority_token_id": 80001, "mode": "validity", "num_shards": 6, "position": 1, "shard_ids": ["shard_0000", "shard_0001", "shard_0002", "shard_0003", "shard_0004", "shard_0005"], "shard_token_ids": [80001, 80001, 80002, 80002, 80003, 80001], "target_id": "export_target_00", "target_prefix_token_ids": [501, 60000, 80000], "target_token_id": 80001, "vote_counts": {"80001": 3, "80002": 2, "80003": 1}}
{"example_id": "export_example_0000", "majority_token_id": 80002, "mode": "validity", "num_shards": 6, "position": 2, "shard_ids": ["shard_0000", "shard_0001", "shard_0002", "shard_0003", "shard_0004", "shard_0005"], "shard_token_ids": [80002, 80002, 80003, 80003, 80004, 80002], "target_id": "export_target_00", "target_prefix_token_ids": [501, 60000, 80000, 80001], "target_token_id": 80002, "vote_counts": {"80002": 3, "80003": 2, "80004": 1}}
```

The validity prefix at position 1 includes target token `80000`; the prefix at position 2 includes target tokens `80000, 80001`. It does not use a clean majority-prefix rollout.

## 7. Example Metadata Sidecars

`outputs/export_mock/stability_votes.meta.json`:

```json
{
  "backend": "mock",
  "created_at": "2026-06-01T14:20:31.900633+00:00",
  "mode": "stability",
  "notes": "Mock stability export; no real model inference. Majority tokens are committed to the next clean prefix.",
  "num_examples": 3,
  "num_positions": 4,
  "num_shards": 6,
  "output_path": "outputs/export_mock/stability_votes.jsonl",
  "schema_version": "vpa-token-votes/v1",
  "shard_ids": ["shard_0000", "shard_0001", "shard_0002", "shard_0003", "shard_0004", "shard_0005"]
}
```

`outputs/export_mock/validity_votes.meta.json`:

```json
{
  "backend": "mock",
  "created_at": "2026-06-01T14:20:37.566533+00:00",
  "mode": "validity",
  "notes": "Mock validity export; no real model inference. Prefixes are extended with target tokens, not clean majority tokens.",
  "num_examples": 3,
  "num_positions": 4,
  "num_shards": 6,
  "output_path": "outputs/export_mock/validity_votes.jsonl",
  "schema_version": "vpa-token-votes/v1",
  "shard_ids": ["shard_0000", "shard_0001", "shard_0002", "shard_0003", "shard_0004", "shard_0005"]
}
```

## 8. Dependency And Import Check

Command run:

```bash
rg -n "^(import|from) (torch|transformers|peft|gurobi|gurobipy)|external\\.VPA|VPA-main" vpa_integration
```

The only matches were existing path/documentation references:

- `vpa_integration/config.py` constructs the default `external/VPA-main` path.
- `vpa_integration/README.md` documents that `external/VPA-main` remains external.

No torch, transformers, PEFT, Gurobi, or VPA-main imports were added.

Final prohibited-import check:

```bash
rg -n "^(import|from) (torch|transformers|peft|gurobi|gurobipy)" vpa_integration
```

No matches.

## 9. external/VPA-main Status

`external/VPA-main` was not modified.

Command run:

```bash
git status --short external/VPA-main
```

Output was empty.

`toy_certificate` was also untouched:

```bash
git status --short toy_certificate
```

Output was empty.

## 10. What Remains For Stage 4

- Add a real VPA backend in a separate runtime module that may import model dependencies.
- Keep schema, IO, metadata, and validation modules dependency-light.
- Wire real adapter discovery and tokenizer/model setup behind the backend abstraction.
- Run only small real-inference smoke exports before scaling.
- Convert validated artifacts into downstream tensor forms for baselines and MILPs.
- Do not call baselines or MILPs inside the exporter.
