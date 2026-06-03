"""Schemas for saved token-level VPA vote artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Literal


def compute_vote_counts(shard_token_ids: list[int]) -> dict[int, int]:
    """Count token votes while preserving integer token ids."""

    return dict(Counter(shard_token_ids))


def compute_majority_token_id(shard_token_ids: list[int]) -> int:
    """Return the plurality token id, breaking ties by first shard order."""

    if not shard_token_ids:
        raise ValueError("shard_token_ids must be non-empty")
    return Counter(shard_token_ids).most_common(1)[0][0]


def _validate_shard_votes(shard_ids: list[str], shard_token_ids: list[int], num_shards: int) -> None:
    if not shard_ids:
        raise ValueError("shard_ids must be non-empty")
    if len(shard_ids) != len(shard_token_ids):
        raise ValueError("shard_ids and shard_token_ids must have the same length")
    if len(shard_ids) != num_shards:
        raise ValueError("num_shards must match the number of shard ids")


@dataclass(frozen=True)
class StabilityVoteRow:
    """One stability-mode token vote row for one example and position."""

    mode: Literal["stability"]
    example_id: str
    position: int
    prefix_token_ids: list[int]
    shard_ids: list[str]
    shard_token_ids: list[int]
    vote_counts: dict[int, int]
    majority_token_id: int
    num_shards: int

    @classmethod
    def from_shard_votes(
        cls,
        *,
        example_id: str,
        position: int,
        prefix_token_ids: list[int],
        shard_ids: list[str],
        shard_token_ids: list[int],
    ) -> "StabilityVoteRow":
        num_shards = len(shard_ids)
        _validate_shard_votes(shard_ids, shard_token_ids, num_shards)
        return cls(
            mode="stability",
            example_id=example_id,
            position=position,
            prefix_token_ids=list(prefix_token_ids),
            shard_ids=list(shard_ids),
            shard_token_ids=list(shard_token_ids),
            vote_counts=compute_vote_counts(shard_token_ids),
            majority_token_id=compute_majority_token_id(shard_token_ids),
            num_shards=num_shards,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ValidityVoteRow:
    """One validity-mode target-prefix token vote row."""

    mode: Literal["validity"]
    example_id: str
    target_id: str
    position: int
    target_prefix_token_ids: list[int]
    target_token_id: int
    shard_ids: list[str]
    shard_token_ids: list[int]
    vote_counts: dict[int, int]
    majority_token_id: int
    num_shards: int

    @classmethod
    def from_shard_votes(
        cls,
        *,
        example_id: str,
        target_id: str,
        position: int,
        target_prefix_token_ids: list[int],
        target_token_id: int,
        shard_ids: list[str],
        shard_token_ids: list[int],
    ) -> "ValidityVoteRow":
        num_shards = len(shard_ids)
        _validate_shard_votes(shard_ids, shard_token_ids, num_shards)
        return cls(
            mode="validity",
            example_id=example_id,
            target_id=target_id,
            position=position,
            target_prefix_token_ids=list(target_prefix_token_ids),
            target_token_id=target_token_id,
            shard_ids=list(shard_ids),
            shard_token_ids=list(shard_token_ids),
            vote_counts=compute_vote_counts(shard_token_ids),
            majority_token_id=compute_majority_token_id(shard_token_ids),
            num_shards=num_shards,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
