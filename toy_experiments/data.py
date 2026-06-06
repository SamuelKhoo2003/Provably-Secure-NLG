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
    distribution: str = "deterministic",
    num_competitor_min: int = 2,
    num_competitor_max: int = 8,
    target_count_min: int = 0,
    target_count_max: int = 3,
    competitor_gap_min: int = 2,
    competitor_gap_max: int = 6,
    competitor_jitter: int = 2,
    row_difficulty_jitter: bool = True,
    position_difficulty_jitter: bool = True,
) -> ToyData:
    """Generate an artificial controlled validity-only stress-test instance.

    This is not a natural language benchmark. The construction intentionally
    creates two effects: many tied non-target competitors make TPA stronger than
    a plain top-vs-target count margin, and mostly different influenced shard
    groups make the shared full-sequence MILP grow with sequence length.
    """
    if distribution == "heterogeneous":
        return _generate_heterogeneous_validity_demo_votes(
            L=L,
            group_size=group_size,
            overlap=overlap,
            N=N,
            T=T,
            seed=seed,
            K=K,
            num_competitor_min=num_competitor_min,
            num_competitor_max=num_competitor_max,
            target_count_min=target_count_min,
            target_count_max=target_count_max,
            competitor_gap_min=competitor_gap_min,
            competitor_gap_max=competitor_gap_max,
            competitor_jitter=competitor_jitter,
            row_difficulty_jitter=row_difficulty_jitter,
            position_difficulty_jitter=position_difficulty_jitter,
        )
    if distribution != "deterministic":
        raise ValueError("validity_demo distribution must be 'deterministic' or 'heterogeneous'")

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


def _generate_heterogeneous_validity_demo_votes(
    *,
    L: int,
    group_size: int,
    overlap: int,
    N: int,
    T: int,
    seed: int,
    K: int | None,
    num_competitor_min: int,
    num_competitor_max: int,
    target_count_min: int,
    target_count_max: int,
    competitor_gap_min: int,
    competitor_gap_max: int,
    competitor_jitter: int,
    row_difficulty_jitter: bool,
    position_difficulty_jitter: bool,
) -> ToyData:
    generator_name = "validity_demo"
    if L < 1 or N < 1:
        raise ValueError("L and N must be at least 1")
    if T < 8:
        raise ValueError("T must be at least 8 for heterogeneous validity_demo instances")
    if group_size < 2:
        raise ValueError("group_size must be at least 2 for heterogeneous validity_demo instances")
    if overlap < 0 or overlap >= group_size:
        raise ValueError("overlap must be in [0, group_size)")
    if num_competitor_min < 1 or num_competitor_max < num_competitor_min:
        raise ValueError("invalid heterogeneous competitor-count range")
    if target_count_min < 0 or target_count_max < target_count_min:
        raise ValueError("invalid heterogeneous target-count range")
    if competitor_gap_min < 1 or competitor_gap_max < competitor_gap_min:
        raise ValueError("invalid heterogeneous competitor-gap range")
    if competitor_gap_max > group_size:
        raise ValueError(
            f"competitor_gap_max={competitor_gap_max} exceeds group_size={group_size}; "
            "the intended shard group may not be able to force every target token"
        )

    stride = group_size - overlap
    min_required_shards = group_size + (L - 1) * stride
    if K is None:
        K = min_required_shards
    elif K < min_required_shards:
        raise ValueError(
            f"{generator_name} requires K >= {min_required_shards} for L={L}, "
            f"group_size={group_size}, overlap={overlap}; got K={K}"
        )
    if T - 1 < num_competitor_min:
        raise ValueError(f"T={T} is too small for num_competitor_min={num_competitor_min}")

    rng = np.random.default_rng(seed)
    all_shards = np.arange(K, dtype=np.int64)
    base_groups = [np.arange(j * stride, j * stride + group_size, dtype=np.int64) for j in range(L)]
    spare_shards = np.arange(min_required_shards, K, dtype=np.int64)

    base_token = np.zeros((N, L), dtype=np.int64)
    for j in range(L):
        base_token[:, j] = j % T
    clean_pred = base_token.copy()
    target = ((base_token + 1) % T).astype(np.int64)
    val_base = ((target + 1) % T).astype(np.int64)

    stab_votes = np.repeat(base_token[None, :, :], K, axis=0).astype(np.int64)
    val_votes = np.empty((K, N, L), dtype=np.int64)
    influence = np.zeros((K, N, L), dtype=np.int64)
    groups_by_cell: list[list[np.ndarray]] = []

    row_offsets = (
        rng.choice(np.array([0, 1, 2, 3], dtype=np.int64), size=N, p=[0.25, 0.35, 0.25, 0.15])
        if row_difficulty_jitter
        else np.zeros(N, dtype=np.int64)
    )
    position_offsets = rng.integers(-1, 2, size=(N, L)) if position_difficulty_jitter else np.zeros((N, L), dtype=np.int64)

    for i in range(N):
        motif_size = int(min(len(spare_shards), max(0, group_size // 2)))
        row_motif = rng.choice(spare_shards, size=motif_size, replace=False) if motif_size else np.array([], dtype=np.int64)
        row_groups: list[np.ndarray] = []
        for j in range(L):
            group = _heterogeneous_validity_group(
                base_group=base_groups[j],
                previous_group=row_groups[-1] if row_groups else None,
                row_motif=row_motif,
                group_size=group_size,
                all_shards=all_shards,
                rng=rng,
            )
            row_groups.append(group)
            influence[group, i, j] = 1

            runner = int((base_token[i, j] + 2) % T)
            non_group = np.array([k for k in all_shards if k not in set(group.tolist())], dtype=np.int64)
            runner_count = min(len(non_group), max(1, (K - 1) // 2))
            stab_votes[non_group[:runner_count], i, j] = runner

            h = int(target[i, j])
            counts = _sample_heterogeneous_validity_counts(
                K=K,
                T=T,
                target=h,
                target_count_min=target_count_min,
                target_count_max=target_count_max,
                num_competitor_min=num_competitor_min,
                num_competitor_max=num_competitor_max,
                competitor_gap_min=competitor_gap_min,
                competitor_gap_max=competitor_gap_max,
                competitor_jitter=competitor_jitter,
                difficulty_offset=int(row_offsets[i] + position_offsets[i, j]),
                group_size=group_size,
                rng=rng,
            )
            val_votes[:, i, j] = _votes_from_counts_with_group_priority(counts, target=h, group=group, rng=rng)
        groups_by_cell.append(row_groups)

    stab_counts = compute_counts(stab_votes, T)
    runner_up = runner_up_tokens(stab_counts, clean_pred)
    val_counts = compute_counts(val_votes, T)
    _check_validity_demo_group_feasibility(
        val_votes=val_votes,
        val_counts=val_counts,
        target=target,
        groups=[groups_by_cell[0][j] for j in range(L)],
        target_gaps=[group_size for _ in range(L)],
        group_size=group_size,
        overlap=overlap,
        generator_name=generator_name,
        groups_by_cell=groups_by_cell,
    )
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


def _heterogeneous_validity_group(
    *,
    base_group: np.ndarray,
    previous_group: np.ndarray | None,
    row_motif: np.ndarray,
    group_size: int,
    all_shards: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    group = list(int(k) for k in base_group)
    if previous_group is not None and rng.random() < 0.35:
        replace_count = int(rng.integers(1, min(3, group_size) + 1))
        shared = rng.choice(previous_group, size=replace_count, replace=False)
        group[-replace_count:] = [int(k) for k in shared]
    if row_motif.size and rng.random() < 0.45:
        group[int(rng.integers(0, group_size))] = int(rng.choice(row_motif))

    unique = []
    for shard in group:
        if shard not in unique:
            unique.append(shard)
    if len(unique) < group_size:
        for shard in all_shards:
            shard_int = int(shard)
            if shard_int not in unique:
                unique.append(shard_int)
                if len(unique) == group_size:
                    break
    return np.array(unique[:group_size], dtype=np.int64)


def _sample_heterogeneous_validity_counts(
    *,
    K: int,
    T: int,
    target: int,
    target_count_min: int,
    target_count_max: int,
    num_competitor_min: int,
    num_competitor_max: int,
    competitor_gap_min: int,
    competitor_gap_max: int,
    competitor_jitter: int,
    difficulty_offset: int,
    group_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    counts = np.zeros(T, dtype=np.int64)
    target_count = int(rng.integers(target_count_min, target_count_max + 1))
    counts[target] = target_count
    competitors = [token for token in range(T) if token != target]
    max_competitors = min(num_competitor_max, len(competitors))
    min_competitors = min(num_competitor_min, max_competitors)
    num_competitors = int(rng.integers(min_competitors, max_competitors + 1))
    if rng.random() < 0.15:
        num_competitors = 1
    active = rng.choice(competitors, size=num_competitors, replace=False)
    gap = int(rng.integers(competitor_gap_min, competitor_gap_max + 1) + difficulty_offset)
    gap = int(np.clip(gap, 1, group_size))
    top_count = max(target_count + gap, int(np.ceil((K - target_count) / max(1, T - 1))))

    for token in active:
        jitter = int(rng.integers(0, competitor_jitter + 1)) if competitor_jitter > 0 else 0
        counts[int(token)] = max(target_count + 1, top_count - jitter)

    remaining = K - int(np.sum(counts))
    if remaining < 0:
        for token in sorted(active, key=lambda t: int(counts[int(t)]), reverse=True):
            if remaining == 0:
                break
            removable = min(int(counts[int(token)] - target_count - 1), -remaining)
            counts[int(token)] -= removable
            remaining += removable
    if remaining < 0:
        raise ValueError("heterogeneous validity_demo sampled counts exceeding K")

    fill_candidates = [token for token in competitors if counts[token] < top_count]
    while remaining > 0 and fill_candidates:
        token = int(rng.choice(fill_candidates))
        counts[token] += 1
        remaining -= 1
        if counts[token] >= top_count:
            fill_candidates = [candidate for candidate in fill_candidates if counts[candidate] < top_count]
    if remaining > 0:
        raise ValueError("heterogeneous validity_demo could not distribute all shard votes")
    return counts


def _votes_from_counts_with_group_priority(
    counts: np.ndarray,
    *,
    target: int,
    group: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    K = int(np.sum(counts))
    votes = np.empty(K, dtype=np.int64)
    token_pool = [token for token, count in enumerate(counts) for _ in range(int(count))]
    non_target_tokens = [token for token in token_pool if token != target]
    highest = sorted(set(non_target_tokens), key=lambda token: (-int(counts[token]), int(token)))
    group_tokens: list[int] = []
    for token in highest:
        while token in non_target_tokens and len(group_tokens) < len(group):
            group_tokens.append(token)
            non_target_tokens.remove(token)
        if len(group_tokens) == len(group):
            break
    if len(group_tokens) < len(group):
        raise ValueError("heterogeneous validity_demo could not assign non-target group votes")
    rng.shuffle(non_target_tokens)
    remaining_tokens = non_target_tokens + [target for _ in range(int(counts[target]))]
    rng.shuffle(remaining_tokens)
    votes[group] = np.array(group_tokens, dtype=np.int64)
    non_group = [k for k in range(K) if k not in set(int(x) for x in group)]
    votes[non_group] = np.array(remaining_tokens, dtype=np.int64)
    return votes


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
    max_non_target_count = max(target_gaps)
    if max_non_target_count > group_size:
        raise ValueError(
            f"{generator_name} requires target_gap <= group_size so each token can be "
            f"forced by its intended shard group; got target_gap={max_non_target_count}, group_size={group_size}"
        )
    min_required_tokens = int(np.ceil(K / max_non_target_count)) + 1
    if T < min_required_tokens:
        raise ValueError(
            f"{generator_name} requires T >= {min_required_tokens} to spread K={K} non-target "
            f"votes with no competitor above target_gap={max_non_target_count}; got T={T}"
        )
    for i in range(N):
        for j in range(L):
            group = groups[j]
            influence[group, i, j] = 1
            runner = int((base_token[i, j] + 2) % T)
            non_group_for_stability = np.array([k for k in all_shards if k not in set(group.tolist())], dtype=np.int64)
            runner_count = min(len(non_group_for_stability), max(1, (K - 1) // 2))
            stab_votes[non_group_for_stability[:runner_count], i, j] = runner

            h = int(target[i, j])
            target_gap = int(target_gaps[j])
            competitors = [token for token in range(T) if token != h]
            if len(competitors) * target_gap < K:
                raise ValueError(
                    f"{generator_name} cannot assign K={K} votes at target_gap={target_gap} "
                    f"without exceeding the per-competitor cap; T={T}"
                )

            votes = np.empty(K, dtype=np.int64)
            shard_order = list(group) + [int(k) for k in all_shards if k not in set(group.tolist())]
            for offset, k in enumerate(shard_order):
                # Every non-target class receives at most target_gap votes. The
                # harmful target receives zero votes before poisoning.
                votes[k] = competitors[(offset // target_gap) % len(competitors)]
            val_votes[:, i, j] = votes

    stab_counts = compute_counts(stab_votes, T)
    runner_up = runner_up_tokens(stab_counts, clean_pred)
    val_counts = compute_counts(val_votes, T)
    _check_validity_demo_group_feasibility(
        val_votes=val_votes,
        val_counts=val_counts,
        target=target,
        groups=groups,
        target_gaps=target_gaps,
        group_size=group_size,
        overlap=overlap,
        generator_name=generator_name,
    )
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


def _check_validity_demo_group_feasibility(
    *,
    val_votes: np.ndarray,
    val_counts: np.ndarray,
    target: np.ndarray,
    groups: list[np.ndarray],
    target_gaps: list[int],
    group_size: int,
    overlap: int,
    generator_name: str,
    groups_by_cell: list[list[np.ndarray]] | None = None,
) -> None:
    """Verify each intended token shard group can force its target locally."""
    K, N, L = val_votes.shape
    T = val_counts.shape[2]
    for i in range(N):
        for j in range(L):
            h = int(target[i, j])
            target_count = int(val_counts[i, j, h])
            group = groups_by_cell[i][j] if groups_by_cell is not None else groups[j]
            for c in range(T):
                if c == h:
                    continue
                required_deficit = int(val_counts[i, j, c] - target_count)
                if required_deficit <= 0:
                    continue
                available_contribution = int(
                    sum(
                        int(val_votes[k, i, j] != h) + int(val_votes[k, i, j] == c)
                        for k in group
                    )
                )
                if available_contribution < required_deficit:
                    raise ValueError(
                        f"{generator_name} generated an infeasible target cell: "
                        f"L={L}, K={K}, j={j}, group_size={group_size}, overlap={overlap}, "
                        f"target_gap={target_gaps[j]}, available_contribution={available_contribution}, "
                        f"required_deficit={required_deficit}"
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

    The shared MILP checks all competitors. Runner-up tokens remain useful for
    the count-only clean winner-vs-runner-up DPA margin.
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
