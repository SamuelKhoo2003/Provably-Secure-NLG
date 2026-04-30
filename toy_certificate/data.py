from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ToyData:
    votes: np.ndarray
    clean_counts: np.ndarray
    clean_pred: np.ndarray
    runner_up: np.ndarray
    target: np.ndarray
    base_token: np.ndarray


def generate_toy_votes(K: int, N: int, L: int, T: int, delta: float = 0.2, seed: int = 0) -> ToyData:
    """Generate toy shard votes and derived clean prediction metadata.

    Returns votes with shape ``[K, N, L]`` and all cell-level quantities needed
    by the MILP solvers.
    """
    _validate_dimensions(K, N, L, T)
    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta must be in [0, 1]")

    rng = np.random.default_rng(seed)
    base_token = rng.integers(0, T, size=(N, L), dtype=np.int64)
    votes = np.empty((K, N, L), dtype=np.int64)

    for k in range(K):
        disagreement = rng.random((N, L)) < delta
        votes[k] = base_token
        if np.any(disagreement):
            alternatives = rng.integers(0, T - 1, size=(N, L), dtype=np.int64)
            alternatives = alternatives + (alternatives >= base_token)
            votes[k, disagreement] = alternatives[disagreement]

    clean_counts = compute_counts(votes, T)
    clean_pred = majority_predictions(clean_counts)
    runner_up = runner_up_tokens(clean_counts, clean_pred)
    target = generate_targets(clean_pred, T, seed=seed + 1)

    return ToyData(
        votes=votes,
        clean_counts=clean_counts,
        clean_pred=clean_pred,
        runner_up=runner_up,
        target=target,
        base_token=base_token,
    )


def compute_counts(votes: np.ndarray, T: int) -> np.ndarray:
    """Return clean counts with shape ``[N, L, T]``."""
    if votes.ndim != 3:
        raise ValueError("votes must have shape [K, N, L]")
    K, N, L = votes.shape
    if T < 2:
        raise ValueError("T must be at least 2")
    if votes.min(initial=0) < 0 or votes.max(initial=0) >= T:
        raise ValueError("votes contain token ids outside [0, T)")

    counts = np.zeros((N, L, T), dtype=np.int64)
    for i in range(N):
        for j in range(L):
            counts[i, j] = np.bincount(votes[:, i, j], minlength=T)
    return counts


def majority_predictions(clean_counts: np.ndarray) -> np.ndarray:
    """Return majority predictions with deterministic smallest-token tie break."""
    if clean_counts.ndim != 3:
        raise ValueError("clean_counts must have shape [N, L, T]")
    return np.argmax(clean_counts, axis=2).astype(np.int64)


def runner_up_tokens(clean_counts: np.ndarray, clean_pred: np.ndarray) -> np.ndarray:
    """Return the second-ranked token per cell with deterministic tie breaking."""
    N, L, T = clean_counts.shape
    runner_up = np.empty((N, L), dtype=np.int64)
    token_ids = np.arange(T)
    for i in range(N):
        for j in range(L):
            winner = int(clean_pred[i, j])
            candidates = token_ids[token_ids != winner]
            # Sort by descending count, then ascending token id.
            ordered = sorted(candidates, key=lambda t: (-int(clean_counts[i, j, t]), int(t)))
            runner_up[i, j] = int(ordered[0])
    return runner_up


def generate_targets(clean_pred: np.ndarray, T: int, seed: int = 0) -> np.ndarray:
    """Return harmful targets with ``target[i, j] != clean_pred[i, j]``."""
    if T < 2:
        raise ValueError("T must be at least 2 to choose non-clean targets")
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, T - 1, size=clean_pred.shape, dtype=np.int64)
    return (raw + (raw >= clean_pred)).astype(np.int64)


def stability_margins(clean_counts: np.ndarray, clean_pred: np.ndarray, runner_up: np.ndarray) -> np.ndarray:
    """Return clean winner-vs-runner-up vote margins per cell."""
    N, L = clean_pred.shape
    margins = np.empty((N, L), dtype=np.int64)
    for i in range(N):
        for j in range(L):
            w = int(clean_pred[i, j])
            r = int(runner_up[i, j])
            margins[i, j] = int(clean_counts[i, j, w] - clean_counts[i, j, r])
    return margins


def _validate_dimensions(K: int, N: int, L: int, T: int) -> None:
    if K < 1:
        raise ValueError("K must be at least 1")
    if N < 1:
        raise ValueError("N must be at least 1")
    if L < 1:
        raise ValueError("L must be at least 1")
    if T < 2:
        raise ValueError("T must be at least 2")
