"""Token vote backend abstractions for VPA export orchestration."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Protocol


Mode = Literal["stability", "validity"]


@dataclass(frozen=True)
class VoteRequest:
    """Context for one next-token vote collection step."""

    mode: Mode
    example_id: str
    position: int
    prefix_token_ids: list[int]
    target_id: str | None = None
    target_token_id: int | None = None


class TokenVoteBackend(Protocol):
    """Minimal interface required by the vote exporter."""

    name: str

    def predict_next_token_for_shards(self, request: VoteRequest, shard_ids: list[str]) -> list[int]:
        """Return one next-token id per shard id, preserving shard order."""


class MockTokenVoteBackend:
    """Deterministic backend used to exercise exporter orchestration.

    This class does not load models or adapters. It derives token ids from the
    request context and shard id so artifact shape and prefix flow can be tested
    before real VPA inference is connected.
    """

    name = "mock"

    def __init__(self, seed: int = 17) -> None:
        self.seed = seed

    def predict_next_token_for_shards(self, request: VoteRequest, shard_ids: list[str]) -> list[int]:
        if request.mode == "stability":
            return [self._stability_vote(request, shard_id) for shard_id in shard_ids]
        if request.mode == "validity":
            return [self._validity_vote(request, shard_id) for shard_id in shard_ids]
        raise ValueError(f"Unsupported vote mode: {request.mode}")

    def _stability_vote(self, request: VoteRequest, shard_id: str) -> int:
        example_idx = _numeric_suffix(request.example_id)
        shard_idx = _numeric_suffix(shard_id)
        base_token = 20_000 + example_idx * 100 + request.position
        runner_up = base_token + 1
        alternate = base_token + 2
        if shard_idx % 4 in {0, 1}:
            return base_token
        if shard_idx % 4 == 2:
            return runner_up
        return alternate if self._coin(request, shard_id, salt=3) else base_token

    def _validity_vote(self, request: VoteRequest, shard_id: str) -> int:
        if request.target_token_id is None:
            raise ValueError("validity vote requests require target_token_id")
        shard_idx = _numeric_suffix(shard_id)
        target_token = request.target_token_id
        competitor = target_token + 1
        alternate = target_token + 2
        if shard_idx % 5 in {0, 1}:
            return target_token
        if shard_idx % 5 in {2, 3}:
            return competitor
        return target_token if self._coin(request, shard_id, salt=5) else alternate

    def _coin(self, request: VoteRequest, shard_id: str, *, salt: int) -> bool:
        seed = (
            self.seed
            + 97 * _numeric_suffix(request.example_id)
            + 31 * request.position
            + 13 * _numeric_suffix(shard_id)
            + salt
        )
        return random.Random(seed).random() < 0.5


def make_backend(name: str, *, seed: int = 17) -> TokenVoteBackend:
    """Construct a backend by name."""

    if name == "mock":
        return MockTokenVoteBackend(seed=seed)
    if name == "vpa":
        raise NotImplementedError("Construct the real VPA backend through export_votes with --enable-real-inference and explicit paths")
    raise ValueError(f"Unsupported backend: {name}")


def _numeric_suffix(value: str) -> int:
    digits = []
    for char in reversed(value):
        if not char.isdigit():
            break
        digits.append(char)
    if not digits:
        return 0
    return int("".join(reversed(digits)))
