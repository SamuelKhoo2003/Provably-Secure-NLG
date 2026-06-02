# VPA Integration Audit - Stage 2

Audit date: 2026-06-01

## 1. Files Created Or Modified

Created:

- `vpa_integration/generate_mock_votes.py`
- `vpa_integration/validate_votes.py`
- `vpa_integration_AUDIT_stage2.md`

Modified:

- `vpa_integration/README.md`

No schema changes were required.

## 2. Mock Vote Generation Command

Command run:

```bash
python3 -B -m vpa_integration.generate_mock_votes --output-dir outputs/mock
```

Output:

```text
Wrote outputs/mock/stability_votes.jsonl
Wrote outputs/mock/validity_votes.jsonl
```

Generated outputs:

- `outputs/mock/stability_votes.jsonl`
- `outputs/mock/validity_votes.jsonl`

## 3. Validation Commands

Commands run:

```bash
python3 -B -m vpa_integration.validate_votes outputs/mock/stability_votes.jsonl
python3 -B -m vpa_integration.validate_votes outputs/mock/validity_votes.jsonl
```

Outputs:

```text
Validation passed for outputs/mock/stability_votes.jsonl: 12 rows
Validation passed for outputs/mock/validity_votes.jsonl: 12 rows
```

Negative-path check run:

```bash
python3 -B -c 'from vpa_integration.validate_votes import VoteValidationError, validate_vote_row ...'
```

Output:

```text
bad_counts: caught: row: vote_counts {10: 1, 11: 1} do not match shard_token_ids counts {10: 2}
bad_lengths: caught: row: len(shard_ids)=2 does not match len(shard_token_ids)=1
```

CLI negative-path check run:

```bash
python3 -B -c 'import json, subprocess, tempfile ...'
```

Output:

```text
bad_counts: returncode=1; Validation failed for /var/folders/.../tmp47zmy5dx.jsonl: row 1: vote_counts {10: 1, 11: 1} do not match shard_token_ids counts {10: 2}
bad_lengths: returncode=1; Validation failed for /var/folders/.../tmpei5cj57p.jsonl: row 1: len(shard_ids)=2 does not match len(shard_token_ids)=1
```

## 4. Validation Result

Passed.

Both generated mock artifacts validated successfully. The validator also caught:

- inconsistent `vote_counts`
- mismatched `shard_ids` / `shard_token_ids` lengths

The CLI returned non-zero status for both invalid temporary artifacts.

## 5. Example Rows

First three rows from `outputs/mock/stability_votes.jsonl`:

```json
{"example_id": "mock_example_0000", "majority_token_id": 2000, "mode": "stability", "num_shards": 6, "position": 0, "prefix_token_ids": [101, 1000], "shard_ids": ["shard_0000", "shard_0001", "shard_0002", "shard_0003", "shard_0004", "shard_0005"], "shard_token_ids": [2000, 2000, 2001, 2002, 2000, 2000], "vote_counts": {"2000": 4, "2001": 1, "2002": 1}}
{"example_id": "mock_example_0000", "majority_token_id": 2001, "mode": "stability", "num_shards": 6, "position": 1, "prefix_token_ids": [101, 1000, 3000], "shard_ids": ["shard_0000", "shard_0001", "shard_0002", "shard_0003", "shard_0004", "shard_0005"], "shard_token_ids": [2001, 2001, 2002, 2003, 2001, 2001], "vote_counts": {"2001": 4, "2002": 1, "2003": 1}}
{"example_id": "mock_example_0000", "majority_token_id": 2002, "mode": "stability", "num_shards": 6, "position": 2, "prefix_token_ids": [101, 1000, 3000, 3001], "shard_ids": ["shard_0000", "shard_0001", "shard_0002", "shard_0003", "shard_0004", "shard_0005"], "shard_token_ids": [2002, 2002, 2003, 2002, 2002, 2002], "vote_counts": {"2002": 5, "2003": 1}}
```

First three rows from `outputs/mock/validity_votes.jsonl`:

```json
{"example_id": "mock_example_0000", "majority_token_id": 8000, "mode": "validity", "num_shards": 6, "position": 0, "shard_ids": ["shard_0000", "shard_0001", "shard_0002", "shard_0003", "shard_0004", "shard_0005"], "shard_token_ids": [8000, 8000, 8001, 8001, 8000, 8000], "target_id": "mock_target_00", "target_prefix_token_ids": [501, 6000, 7000], "target_token_id": 8000, "vote_counts": {"8000": 4, "8001": 2}}
{"example_id": "mock_example_0000", "majority_token_id": 8001, "mode": "validity", "num_shards": 6, "position": 1, "shard_ids": ["shard_0000", "shard_0001", "shard_0002", "shard_0003", "shard_0004", "shard_0005"], "shard_token_ids": [8001, 8001, 8002, 8002, 8001, 8001], "target_id": "mock_target_00", "target_prefix_token_ids": [501, 6000, 7000, 7901], "target_token_id": 8001, "vote_counts": {"8001": 4, "8002": 2}}
{"example_id": "mock_example_0000", "majority_token_id": 8002, "mode": "validity", "num_shards": 6, "position": 2, "shard_ids": ["shard_0000", "shard_0001", "shard_0002", "shard_0003", "shard_0004", "shard_0005"], "shard_token_ids": [8002, 8002, 8003, 8003, 8004, 8002], "target_id": "mock_target_00", "target_prefix_token_ids": [501, 6000, 7000, 7902, 7903], "target_token_id": 8002, "vote_counts": {"8002": 3, "8003": 2, "8004": 1}}
```

## 6. Assumptions Made

- Mock token ids are synthetic integer tokenizer ids with no relationship to a real tokenizer.
- `position` is zero-based.
- Ties are resolved by Python `Counter(...).most_common(1)` behavior, which follows first occurrence in shard order for equal counts.
- `vote_counts` may have string keys after JSON round trip; validation normalizes keys back to integers before comparison.
- JSONL remains the debug-first format for this stage.

## 7. Schema Changes

None.

The existing Stage 1 schemas already preserve:

- `shard_ids`
- `shard_token_ids`
- `vote_counts`
- `majority_token_id`
- mode-specific prefix and target fields

## 8. external/VPA-main Status

`external/VPA-main` was not modified.

## 9. What Remains For Stage 3

- Add a real but small exporter skeleton that can call VPA adapter inference in a controlled smoke mode.
- Keep model imports isolated to exporter/runtime modules, not schema or IO modules.
- Preserve stable example identities and shard ordering in exported metadata.
- Convert validated artifacts into tensor shapes expected by downstream baseline/MILP code.
- Add negative validation fixtures or tests once the validation behavior is finalized.
