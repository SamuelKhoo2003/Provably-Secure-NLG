"""Validate VPA token vote JSONL artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .io import read_jsonl
from .schemas import compute_majority_token_id, compute_vote_counts


class VoteValidationError(ValueError):
    """Raised when a vote artifact row fails schema validation."""


def validate_vote_file(path: str | Path) -> int:
    """Validate all rows in a vote JSONL file and return the row count."""

    rows = read_jsonl(path)
    for row_idx, row in enumerate(rows, start=1):
        validate_vote_row(row, row_idx=row_idx)
    return len(rows)


def validate_vote_row(row: dict[str, Any], *, row_idx: int | None = None) -> None:
    """Validate one stability or validity vote row."""

    label = f"row {row_idx}" if row_idx is not None else "row"
    mode = row.get("mode")
    if mode not in {"stability", "validity"}:
        raise VoteValidationError(f"{label}: mode must be 'stability' or 'validity', got {mode!r}")

    position = row.get("position")
    if not isinstance(position, int) or position < 0:
        raise VoteValidationError(f"{label}: position must be a non-negative integer")

    shard_ids = row.get("shard_ids")
    shard_token_ids = row.get("shard_token_ids")
    if not isinstance(shard_ids, list):
        raise VoteValidationError(f"{label}: shard_ids must be a list")
    if not isinstance(shard_token_ids, list):
        raise VoteValidationError(f"{label}: shard_token_ids must be a list")
    if len(shard_ids) != len(shard_token_ids):
        raise VoteValidationError(
            f"{label}: len(shard_ids)={len(shard_ids)} does not match len(shard_token_ids)={len(shard_token_ids)}"
        )
    if not shard_ids:
        raise VoteValidationError(f"{label}: shard_ids must be non-empty")
    if not all(isinstance(shard_id, str) for shard_id in shard_ids):
        raise VoteValidationError(f"{label}: every shard id must be a string")
    if not all(_is_int_token(token_id) for token_id in shard_token_ids):
        raise VoteValidationError(f"{label}: every shard token id must be an integer")

    num_shards = row.get("num_shards")
    if not isinstance(num_shards, int):
        raise VoteValidationError(f"{label}: num_shards must be an integer")
    if num_shards != len(shard_ids):
        raise VoteValidationError(f"{label}: num_shards={num_shards} does not match len(shard_ids)={len(shard_ids)}")

    expected_counts = compute_vote_counts(shard_token_ids)
    actual_counts = _normalize_vote_counts(row.get("vote_counts"), label)
    if actual_counts != expected_counts:
        raise VoteValidationError(f"{label}: vote_counts {actual_counts} do not match shard_token_ids counts {expected_counts}")

    majority_token_id = row.get("majority_token_id")
    if not _is_int_token(majority_token_id):
        raise VoteValidationError(f"{label}: majority_token_id must be an integer")
    expected_majority = compute_majority_token_id(shard_token_ids)
    if majority_token_id != expected_majority:
        raise VoteValidationError(
            f"{label}: majority_token_id={majority_token_id} does not match first-occurrence tie rule majority {expected_majority}"
        )

    if mode == "stability":
        _validate_token_list(row.get("prefix_token_ids"), f"{label}: prefix_token_ids")
    else:
        target_id = row.get("target_id")
        if not isinstance(target_id, str) or not target_id:
            raise VoteValidationError(f"{label}: target_id must be a non-empty string")
        _validate_token_list(row.get("target_prefix_token_ids"), f"{label}: target_prefix_token_ids")
        if not _is_int_token(row.get("target_token_id")):
            raise VoteValidationError(f"{label}: target_token_id must be an integer")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a VPA token vote JSONL artifact.")
    parser.add_argument("path", type=Path, help="Path to stability or validity vote JSONL")
    args = parser.parse_args(argv)

    try:
        row_count = validate_vote_file(args.path)
    except (OSError, ValueError) as exc:
        print(f"Validation failed for {args.path}: {exc}")
        return 1

    print(f"Validation passed for {args.path}: {row_count} rows")
    return 0


def _normalize_vote_counts(value: Any, label: str) -> dict[int, int]:
    if not isinstance(value, dict):
        raise VoteValidationError(f"{label}: vote_counts must be an object")
    counts: dict[int, int] = {}
    for raw_key, raw_count in value.items():
        try:
            token_id = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise VoteValidationError(f"{label}: vote_counts key {raw_key!r} is not an integer token id") from exc
        if not isinstance(raw_count, int) or raw_count < 0:
            raise VoteValidationError(f"{label}: vote_counts[{raw_key!r}] must be a non-negative integer")
        counts[token_id] = raw_count
    return counts


def _validate_token_list(value: Any, field_label: str) -> None:
    if not isinstance(value, list):
        raise VoteValidationError(f"{field_label} must be a list")
    if not all(_is_int_token(token_id) for token_id in value):
        raise VoteValidationError(f"{field_label} must contain only integer token ids")


def _is_int_token(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


if __name__ == "__main__":
    raise SystemExit(main())
