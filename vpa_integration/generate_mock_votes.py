"""Generate deterministic mock VPA token vote artifacts."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from .io import write_jsonl
from .schemas import StabilityVoteRow, ValidityVoteRow


def generate_stability_rows(
    *,
    num_examples: int,
    num_positions: int,
    num_shards: int,
    seed: int,
) -> list[StabilityVoteRow]:
    """Create deterministic stability vote rows with shard-level votes."""

    rng = random.Random(seed)
    shard_ids = _shard_ids(num_shards)
    rows: list[StabilityVoteRow] = []
    for example_idx in range(num_examples):
        base_prefix = [101, 1000 + example_idx]
        for position in range(num_positions):
            majority_token = 2000 + example_idx * 100 + position
            runner_up = majority_token + 1
            alternate = majority_token + 2
            shard_token_ids = []
            for shard_idx in range(num_shards):
                if shard_idx % 4 in {0, 1}:
                    token_id = majority_token
                elif shard_idx % 4 == 2:
                    token_id = runner_up
                else:
                    token_id = alternate if rng.random() < 0.35 else majority_token
                shard_token_ids.append(token_id)
            rows.append(
                StabilityVoteRow.from_shard_votes(
                    example_id=f"mock_example_{example_idx:04d}",
                    position=position,
                    prefix_token_ids=base_prefix + [3000 + offset for offset in range(position)],
                    shard_ids=shard_ids,
                    shard_token_ids=shard_token_ids,
                )
            )
    return rows


def generate_validity_rows(
    *,
    num_examples: int,
    num_positions: int,
    num_shards: int,
    seed: int,
) -> list[ValidityVoteRow]:
    """Create deterministic validity vote rows with target-prefix metadata."""

    rng = random.Random(seed + 10_000)
    shard_ids = _shard_ids(num_shards)
    rows: list[ValidityVoteRow] = []
    for example_idx in range(num_examples):
        for target_idx in range(1):
            target_id = f"mock_target_{target_idx:02d}"
            target_prefix = [501, 6000 + example_idx, 7000 + target_idx]
            for position in range(num_positions):
                target_token_id = 8000 + example_idx * 100 + target_idx * 10 + position
                competitor = target_token_id + 1
                other = target_token_id + 2
                shard_token_ids = []
                for shard_idx in range(num_shards):
                    if shard_idx % 5 in {0, 1}:
                        token_id = target_token_id
                    elif shard_idx % 5 in {2, 3}:
                        token_id = competitor
                    else:
                        token_id = target_token_id if rng.random() < 0.5 else other
                    shard_token_ids.append(token_id)
                rows.append(
                    ValidityVoteRow.from_shard_votes(
                        example_id=f"mock_example_{example_idx:04d}",
                        target_id=target_id,
                        position=position,
                        target_prefix_token_ids=target_prefix + [target_token_id - 100 + offset for offset in range(position)],
                        target_token_id=target_token_id,
                        shard_ids=shard_ids,
                        shard_token_ids=shard_token_ids,
                    )
                )
    return rows


def write_mock_votes(
    output_dir: str | Path,
    *,
    num_examples: int = 3,
    num_positions: int = 4,
    num_shards: int = 6,
    seed: int = 7,
) -> tuple[Path, Path]:
    """Generate and write stability and validity mock artifacts."""

    if num_examples < 1:
        raise ValueError("num_examples must be at least 1")
    if num_positions < 1:
        raise ValueError("num_positions must be at least 1")
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")

    output_path = Path(output_dir)
    stability_path = output_path / "stability_votes.jsonl"
    validity_path = output_path / "validity_votes.jsonl"
    write_jsonl(
        stability_path,
        generate_stability_rows(
            num_examples=num_examples,
            num_positions=num_positions,
            num_shards=num_shards,
            seed=seed,
        ),
    )
    write_jsonl(
        validity_path,
        generate_validity_rows(
            num_examples=num_examples,
            num_positions=num_positions,
            num_shards=num_shards,
            seed=seed,
        ),
    )
    return stability_path, validity_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic mock VPA token vote artifacts.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for stability_votes.jsonl and validity_votes.jsonl")
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--num-positions", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    stability_path, validity_path = write_mock_votes(
        args.output_dir,
        num_examples=args.num_examples,
        num_positions=args.num_positions,
        num_shards=args.num_shards,
        seed=args.seed,
    )
    print(f"Wrote {stability_path}")
    print(f"Wrote {validity_path}")
    return 0


def _shard_ids(num_shards: int) -> list[str]:
    return [f"shard_{idx:04d}" for idx in range(num_shards)]


if __name__ == "__main__":
    raise SystemExit(main())
