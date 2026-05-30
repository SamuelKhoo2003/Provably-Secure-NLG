"""Toy vote generation utilities.

This module creates synthetic shard-level token votes for the row/column
poisoning certificate experiments. It returns :class:`ToyData` objects containing
stability votes, validity votes, token counts, clean predictions, harmful
targets, and influence masks. All tensors use shard/prompt/token-position
ordering: votes have shape ``(K, N, L)`` and count tensors have shape
``(N, L, T)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ToyData:
    """Container for one synthetic toy certificate instance.

    ``stab_votes`` are clean-prefix votes used for stability objectives, where
    the attacker changes output away from ``clean_pred``. ``val_votes`` are
    harmful-prefix votes used for validity objectives, where the attacker forces
    ``target``. ``influence`` has shape ``(K, N, L)`` and marks which shard votes
    can affect each prompt row and token position.
    """

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
    """Generate shard votes, counts, targets, and influence masks.

    ``K`` is the number of shards, ``N`` prompt rows, ``L`` token positions, and
    ``T`` vocabulary tokens. Stability votes use the clean autoregressive prefix.
    Validity votes use a harmful-target prefix and give target tokens
    controllable support through ``target_bias``.
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


def generate_validity_demo_votes(
    L: int,
    group_size: int,
    target_gap: int,
    overlap: int = 0,
    N: int = 1,
    T: int = 4,
    seed: int = 0,
    K: int | None = None,
) -> ToyData:
    """Generate an artificial controlled validity-demo instance.

    Each token position has a cheap harmful-target attack supported by its own
    shard group. Groups are disjoint when ``overlap=0`` and share adjacent
    boundary shards when ``overlap>0``. Token-level counts therefore make each
    target look individually cheap, while a full harmful sequence requires a
    common poisoned-shard allocation spanning several mostly different groups.
    """
    gap_pattern = [target_gap for _ in range(L)]
    return _generate_validity_demo_votes_from_gaps(
        L=L,
        group_size=group_size,
        target_gaps=gap_pattern,
        overlap=overlap,
        N=N,
        T=T,
        seed=seed,
        K=K,
        generator_name="validity_demo",
    )


def _generate_validity_demo_votes_from_gaps(
    L: int,
    group_size: int,
    target_gaps: list[int],
    overlap: int = 0,
    N: int = 1,
    T: int = 4,
    seed: int = 0,
    K: int | None = None,
    generator_name: str = "validity_demo",
) -> ToyData:
    if L < 1:
        raise ValueError("L must be at least 1")
    if N < 1:
        raise ValueError("N must be at least 1")
    if T < 4:
        raise ValueError(f"T must be at least 4 for {generator_name} instances")
    if group_size < 1:
        raise ValueError("group_size must be at least 1")
    if len(target_gaps) != L:
        raise ValueError("target_gaps must have length L")
    if any(gap < 1 for gap in target_gaps):
        raise ValueError("all target gaps must be at least 1")
    if overlap < 0 or overlap >= group_size:
        raise ValueError("overlap must be in [0, group_size)")

    stride = group_size - overlap
    min_required_shards = group_size + (L - 1) * stride
    if K is None:
        K = min_required_shards
    elif K < min_required_shards:
        raise ValueError(
            f"{generator_name} requires K >= {min_required_shards} for L={L}, "
            f"group_size={group_size}, overlap={overlap}; got K={K}"
        )
    rng = np.random.default_rng(seed)
    groups = [np.arange(j * stride, j * stride + group_size, dtype=np.int64) for j in range(L)]

    base_token = np.zeros((N, L), dtype=np.int64)
    for j in range(L):
        base_token[:, j] = j % T
    clean_pred = base_token.copy()
    target = ((base_token + 1) % T).astype(np.int64)
    val_base = ((target + 1) % T).astype(np.int64)

    stab_votes = np.repeat(base_token[None, :, :], K, axis=0).astype(np.int64)
    val_votes = np.empty((K, N, L), dtype=np.int64)
    influence = np.zeros((K, N, L), dtype=np.int64)
    all_shards = np.arange(K, dtype=np.int64)
    for i in range(N):
        for j in range(L):
            group = groups[j]
            influence[group, i, j] = 1
            runner = int((base_token[i, j] + 2) % T)
            non_group_for_stability = np.array([k for k in all_shards if k not in set(group.tolist())], dtype=np.int64)
            runner_count = min(len(non_group_for_stability), max(1, (K - 1) // 2))
            stab_votes[non_group_for_stability[:runner_count], i, j] = runner

            h = int(target[i, j])
            main_competitor = int(val_base[i, j])
            other_tokens = [token for token in range(T) if token not in {h, main_competitor}]
            target_gap = int(target_gaps[j])

            target_count = int(np.ceil((K - target_gap) / T))
            main_count = min(K - target_count, target_count + target_gap)
            remaining = K - target_count - main_count

            votes = np.empty(K, dtype=np.int64)
            votes[:] = other_tokens[0]
            ordered_group = group.copy()
            rng.shuffle(ordered_group)
            main_group = ordered_group[: min(group_size, main_count)]
            votes[main_group] = main_competitor

            non_group = np.array([k for k in all_shards if k not in set(group.tolist())], dtype=np.int64)
            rng.shuffle(non_group)
            cursor = 0
            remaining_main = main_count - len(main_group)
            if remaining_main > 0:
                votes[non_group[cursor : cursor + remaining_main]] = main_competitor
                cursor += remaining_main
            votes[non_group[cursor : cursor + target_count]] = h
            cursor += target_count
            for offset, k in enumerate(non_group[cursor : cursor + remaining]):
                votes[k] = other_tokens[offset % len(other_tokens)]
            val_votes[:, i, j] = votes

    stab_counts = compute_counts(stab_votes, T)
    runner_up = runner_up_tokens(stab_counts, clean_pred)
    val_counts = compute_counts(val_votes, T)
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
    """Count token votes per prompt row and token position.

    Args:
        votes: Integer token ids with shape ``(K, N, L)``.
        T: Vocabulary size.

    Returns:
        Count tensor with shape ``(N, L, T)``.
    """
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
    """Return majority token predictions with smallest-token tie breaking."""
    if clean_counts.ndim != 3:
        raise ValueError("clean_counts must have shape [N, L, T]")
    return np.argmax(clean_counts, axis=2).astype(np.int64)


def runner_up_tokens(clean_counts: np.ndarray, clean_pred: np.ndarray) -> np.ndarray:
    """Return second-ranked tokens for DPA weakest-token baselines.

    The shared MILP can check all competitors, but runner-up tokens remain useful
    for clean winner-vs-runner-up margins and the runner-up approximation mode.
    """
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
    """Sample harmful targets with ``target[i, j] != clean_pred[i, j]``."""
    if T < 2:
        raise ValueError("T must be at least 2 to choose non-clean targets")
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, T - 1, size=clean_pred.shape, dtype=np.int64)
    return (raw + (raw >= clean_pred)).astype(np.int64)


def generate_influence(K: int, N: int, L: int, mode: str = "dense", seed: int = 0) -> np.ndarray:
    """Generate the poisoned-shard influence mask with shape ``(K, N, L)``.

    ``dense`` allows every poisoned shard to affect every cell, while
    ``row-local`` and ``column-local`` restrict influence by prompt row or token
    position.
    """
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
    """Return clean winner-vs-runner-up vote margins for each cell."""
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
