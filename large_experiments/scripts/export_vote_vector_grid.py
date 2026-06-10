#!/usr/bin/env python3
"""
Export full-scale VPA vote-vector JSONL files into CSV grids.

This script builds the prompt-token grid used by the row-column MILPs.

Important:
- 1b_full and 1b_last3_lora are different clean configurations.
- They are not clean vs poisoned.
- The "target" grid is a hypothetical adversarial target constructed from
  observed non-majority alternative tool-call predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from large_experiments.vote_vector_utils import (
    dpa_certified_radius,
    sorted_counter_items,
)


def dpa_radius_from_counts(counts: Counter[int]) -> int:
    """Compute conservative DPA stability from the two largest token counts."""
    top = sorted_counter_items(counts)[:2]
    if not top:
        return 0

    winner_votes = top[0][1]
    runner_up_votes = top[1][1] if len(top) > 1 else 0
    return dpa_certified_radius(winner_votes, runner_up_votes)


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows and validate the stored final-tool vote metadata."""
    rows: list[dict[str, Any]] = []

    with path.open() as f:
        for line_idx, line in enumerate(f):
            row = json.loads(line)

            required = [
                "vote_vector",
                "token_vote_matrix",
                "vote_counts",
                "majority",
            ]
            missing = [k for k in required if k not in row]
            if missing:
                raise ValueError(f"Row {line_idx} is missing keys: {missing}")

            vote_vector = row["vote_vector"]
            matrix = row["token_vote_matrix"]

            if len(vote_vector) != len(matrix):
                raise ValueError(
                    f"Row {line_idx}: len(vote_vector)={len(vote_vector)} "
                    f"but len(token_vote_matrix)={len(matrix)}"
                )

            lengths = {len(seq) for seq in matrix}
            if len(lengths) != 1:
                raise ValueError(
                    f"Row {line_idx}: shard token lengths are inconsistent: {lengths}"
                )

            stored_counts = dict(row["vote_counts"])
            actual_counts = dict(Counter(vote_vector))
            if stored_counts != actual_counts:
                raise ValueError(
                    f"Row {line_idx}: vote_counts does not match Counter(vote_vector). "
                    f"stored={stored_counts}, actual={actual_counts}"
                )

            row["_line_idx"] = line_idx
            row["_token_len"] = next(iter(lengths))
            rows.append(row)

    return rows


def export_clean_grid(rows: list[dict[str, Any]], horizon: int, out_path: Path) -> None:
    """
    One row per prompt-token cell.

    clean_token_id is the majority token at that cell.
    dpa_stability_radius is top token vs strongest competitor.
    """
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_index",
                "original_row_index",
                "position",
                "clean_token_id",
                "clean_votes",
                "runner_up_token_id",
                "runner_up_votes",
                "num_unique_tokens",
                "dpa_stability_radius",
            ],
        )
        writer.writeheader()

        for prompt_index, row in enumerate(rows):
            matrix = row["token_vote_matrix"]
            k_count = len(matrix)

            for pos in range(horizon):
                counts = Counter(matrix[k][pos] for k in range(k_count))
                top = sorted_counter_items(counts)

                clean_token_id, clean_votes = top[0]
                if len(top) > 1:
                    runner_up_token_id, runner_up_votes = top[1]
                else:
                    runner_up_token_id, runner_up_votes = "", 0

                writer.writerow(
                    {
                        "prompt_index": prompt_index,
                        "original_row_index": row["_line_idx"],
                        "position": pos,
                        "clean_token_id": clean_token_id,
                        "clean_votes": clean_votes,
                        "runner_up_token_id": runner_up_token_id,
                        "runner_up_votes": runner_up_votes,
                        "num_unique_tokens": len(counts),
                        "dpa_stability_radius": dpa_radius_from_counts(counts),
                    }
                )


def export_target_grid(rows: list[dict[str, Any]], horizon: int, out_path: Path) -> None:
    """
    One row per prompt-target-position.

    Target classes are observed non-majority tool calls from vote_vector.
    For each target class, we choose the first shard that voted for that class
    and use its generated token sequence as a representative target sequence.

    active_position = 1 means target_token_id differs from the clean majority token.
    Validity MILPs should normally use only active positions.
    """
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_index",
                "original_row_index",
                "target_class",
                "target_rank_by_votes",
                "target_class_votes",
                "representative_shard_index",
                "position",
                "clean_token_id",
                "clean_votes",
                "target_token_id",
                "target_votes",
                "active_position",
                "dpa_target_radius",
            ],
        )
        writer.writeheader()

        for prompt_index, row in enumerate(rows):
            matrix = row["token_vote_matrix"]
            vote_vector = row["vote_vector"]
            clean_class = row["majority"]
            class_counts = Counter(vote_vector)

            target_classes = [
                cls
                for cls, _ in sorted_counter_items(class_counts)
                if cls != clean_class
            ]

            for target_rank, target_class in enumerate(target_classes, start=1):
                representative_shards = [
                    k for k, pred in enumerate(vote_vector) if pred == target_class
                ]
                if not representative_shards:
                    continue

                rep_k = representative_shards[0]
                target_tokens = matrix[rep_k][:horizon]

                for pos in range(horizon):
                    token_counts = Counter(matrix[k][pos] for k in range(len(matrix)))
                    token_top = sorted_counter_items(token_counts)
                    clean_token_id, clean_votes = token_top[0]

                    target_token_id = target_tokens[pos]
                    target_votes = token_counts.get(target_token_id, 0)
                    active = int(target_token_id != clean_token_id)

                    # DPA target diagnostic, clean majority versus this target token.
                    dpa_target_radius = max(
                        0, (clean_votes - target_votes - 1) // 2
                    )

                    writer.writerow(
                        {
                            "prompt_index": prompt_index,
                            "original_row_index": row["_line_idx"],
                            "target_class": target_class,
                            "target_rank_by_votes": target_rank,
                            "target_class_votes": class_counts[target_class],
                            "representative_shard_index": rep_k,
                            "position": pos,
                            "clean_token_id": clean_token_id,
                            "clean_votes": clean_votes,
                            "target_token_id": target_token_id,
                            "target_votes": target_votes,
                            "active_position": active,
                            "dpa_target_radius": dpa_target_radius,
                        }
                    )


def export_aggregate_tool_votes(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Write one row per observed final-tool class and prompt."""
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_index",
                "original_row_index",
                "tool_call_class",
                "votes",
                "is_majority",
                "rank_by_votes",
                "stored_majority",
                "stored_robustness_radius",
                "ground_truth_correct",
                "majority_is_safe",
            ],
        )
        writer.writeheader()

        for prompt_index, row in enumerate(rows):
            counts = Counter(row["vote_vector"])
            majority = row["majority"]

            for rank, (cls, votes) in enumerate(sorted_counter_items(counts), start=1):
                writer.writerow(
                    {
                        "prompt_index": prompt_index,
                        "original_row_index": row["_line_idx"],
                        "tool_call_class": cls,
                        "votes": votes,
                        "is_majority": int(cls == majority),
                        "rank_by_votes": rank,
                        "stored_majority": majority,
                        "stored_robustness_radius": row.get("robustness_radius", ""),
                        "ground_truth_correct": row.get("ground_truth_correct", ""),
                        "majority_is_safe": row.get("majority_is_safe", ""),
                    }
                )


def export_shard_votes_long(
    rows: list[dict[str, Any]], horizon: int, out_path: Path
) -> None:
    """
    Full shard-aware grid.

    This can be large.
    For H=20 and N=110, it writes 110 * 20 * 500 = 1.1 million rows.
    """
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_index",
                "original_row_index",
                "position",
                "shard_index",
                "shard_name",
                "tool_call_class",
                "token_id",
            ],
        )
        writer.writeheader()

        for prompt_index, row in enumerate(rows):
            matrix = row["token_vote_matrix"]
            vote_vector = row["vote_vector"]
            shards = row.get("shards", [f"shard_{k:04d}" for k in range(len(matrix))])

            for pos in range(horizon):
                for k in range(len(matrix)):
                    writer.writerow(
                        {
                            "prompt_index": prompt_index,
                            "original_row_index": row["_line_idx"],
                            "position": pos,
                            "shard_index": k,
                            "shard_name": shards[k],
                            "tool_call_class": vote_vector[k],
                            "token_id": matrix[k][pos],
                        }
                    )


def export_summary(
    rows_all: list[dict[str, Any]],
    rows_kept: list[dict[str, Any]],
    horizon: int,
    out_path: Path,
    input_path: Path,
    config_name: str,
) -> None:
    """Write high-level dimensions and filtering metadata for an export."""
    lengths = [row["_token_len"] for row in rows_all]

    summary = {
        "config_name": config_name,
        "input_path": str(input_path),
        "horizon": horizon,
        "num_total_prompts": len(rows_all),
        "num_retained_prompts": len(rows_kept),
        "num_dropped_prompts": len(rows_all) - len(rows_kept),
        "num_shards": len(rows_all[0]["vote_vector"]) if rows_all else 0,
        "min_generation_length": min(lengths) if lengths else None,
        "max_generation_length": max(lengths) if lengths else None,
        "target_mode": "observed_alternative_tool_calls",
        "note": (
            "This export is not clean versus poisoned. "
            "It is clean shard outputs plus representative alternative tool-call targets."
        ),
    }

    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--write-shard-grid",
        action="store_true",
        help="Write full long-form shard grid. This may be large.",
    )
    args = parser.parse_args()

    rows_all = load_rows(args.input)
    rows_kept = [
        row for row in rows_all if row["_token_len"] >= args.horizon
    ]

    if not rows_kept:
        raise ValueError(
            f"No rows retained for horizon {args.horizon}. "
            f"Try a smaller horizon."
        )

    out_dir = args.output_dir / args.name / f"H{args.horizon:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    export_clean_grid(
        rows_kept,
        args.horizon,
        out_dir / "clean_grid.csv",
    )
    export_target_grid(
        rows_kept,
        args.horizon,
        out_dir / "target_grid.csv",
    )
    export_aggregate_tool_votes(
        rows_all,
        out_dir / "aggregate_tool_votes.csv",
    )
    export_summary(
        rows_all,
        rows_kept,
        args.horizon,
        out_dir / "summary.json",
        args.input,
        args.name,
    )

    if args.write_shard_grid:
        export_shard_votes_long(
            rows_kept,
            args.horizon,
            out_dir / "shard_votes_long.csv",
        )

    print(f"Wrote outputs to {out_dir}")
    print(f"Retained {len(rows_kept)}/{len(rows_all)} prompts for H={args.horizon}")
    print("Files:")
    print(f"  {out_dir / 'clean_grid.csv'}")
    print(f"  {out_dir / 'target_grid.csv'}")
    print(f"  {out_dir / 'aggregate_tool_votes.csv'}")
    print(f"  {out_dir / 'summary.json'}")
    if args.write_shard_grid:
        print(f"  {out_dir / 'shard_votes_long.csv'}")


if __name__ == "__main__":
    main()
