from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ToyData:
    stab_votes: np.ndarray
    val_votes: np.ndarray
    stab_counts: np.ndarray
    val_counts: np.ndarray
    clean_pred: np.ndarray
    runner_up: np.ndarray
    target: np.ndarray
    base_token: np.ndarray
    val_base: np.ndarray
    influence: np.ndarray

    @property
    def votes(self) -> np.ndarray:
        return self.stab_votes

    @property
    def clean_counts(self) -> np.ndarray:
        return self.stab_counts


def generate_toy_votes(
    K: int,
    N: int,
    L: int,
    T: int,
    delta_stab: float = 0.2,
    delta_val: float = 0.2,
    target_bias: float = 0.2,
    seed: int = 0,
    influence_mode: str = "dense",
) -> ToyData:
    """Generate toy shard votes and derived clean prediction metadata.

    Stability votes use the clean autoregressive prefix. Validity votes use the
    harmful target prefix and give the target token controllable support.
    """
    _validate_dimensions(K, N, L, T)
    for name, value in {"delta_stab": delta_stab, "delta_val": delta_val, "target_bias": target_bias}.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    rng = np.random.default_rng(seed)
    base_token = rng.integers(0, T, size=(N, L), dtype=np.int64)
    stab_votes = np.empty((K, N, L), dtype=np.int64)

    for k in range(K):
        disagreement = rng.random((N, L)) < delta_stab
        stab_votes[k] = base_token
        if np.any(disagreement):
            alternatives = rng.integers(0, T - 1, size=(N, L), dtype=np.int64)
            alternatives = alternatives + (alternatives >= base_token)
            stab_votes[k, disagreement] = alternatives[disagreement]

    stab_counts = compute_counts(stab_votes, T)
    clean_pred = majority_predictions(stab_counts)
    runner_up = runner_up_tokens(stab_counts, clean_pred)
    target = generate_targets(clean_pred, T, seed=seed + 1)

    val_base = _generate_non_target_base(target, T, rng)
    val_votes = np.empty((K, N, L), dtype=np.int64)
    for k in range(K):
        follows_target = rng.random((N, L)) < target_bias
        disagreement = rng.random((N, L)) < delta_val
        val_votes[k] = val_base
        noisy_alternatives = rng.integers(0, T - 1, size=(N, L), dtype=np.int64)
        noisy_alternatives = noisy_alternatives + (noisy_alternatives >= val_base)
        val_votes[k, disagreement] = noisy_alternatives[disagreement]
        val_votes[k, follows_target] = target[follows_target]

    val_counts = compute_counts(val_votes, T)
    influence = generate_influence(K, N, L, mode=influence_mode, seed=seed + 2)

    return ToyData(
        stab_votes=stab_votes,
        val_votes=val_votes,
        stab_counts=stab_counts,
        val_counts=val_counts,
        clean_pred=clean_pred,
        runner_up=runner_up,
        target=target,
        base_token=base_token,
        val_base=val_base,
        influence=influence,
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


def generate_influence(K: int, N: int, L: int, mode: str = "dense", seed: int = 0) -> np.ndarray:
    """Return influence mask with shape ``[K, N, L]``."""
    rng = np.random.default_rng(seed)
    if mode == "dense":
        return np.ones((K, N, L), dtype=np.int64)
    if mode == "row-local":
        row_mask = rng.integers(0, 2, size=(K, N, 1), dtype=np.int64)
        return np.repeat(row_mask, L, axis=2)
    if mode == "column-local":
        col_mask = rng.integers(0, 2, size=(K, 1, L), dtype=np.int64)
        return np.repeat(col_mask, N, axis=1)
    raise ValueError(f"Unknown influence mode: {mode}")


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


def _generate_non_target_base(target: np.ndarray, T: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.integers(0, T - 1, size=target.shape, dtype=np.int64)
    return (raw + (raw >= target)).astype(np.int64)


def _validate_dimensions(K: int, N: int, L: int, T: int) -> None:
    if K < 1:
        raise ValueError("K must be at least 1")
    if N < 1:
        raise ValueError("N must be at least 1")
    if L < 1:
        raise ValueError("L must be at least 1")
    if T < 2:
        raise ValueError("T must be at least 2")
