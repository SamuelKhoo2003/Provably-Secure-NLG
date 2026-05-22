"""Reference baseline computations for toy certificate benchmarks."""

from __future__ import annotations

import numpy as np

from .data import ToyData, stability_margins


def compute_reference_baselines(data: ToyData) -> dict[str, int | float]:
    """Compute baseline columns for the report-facing taxonomy."""
    stability_cell_budgets = cell_stability_budgets(data)
    validity_cell_budgets = cell_validity_budgets(data)
    targeted_validity_cell_budgets = targeted_validity_token_budgets(data)
    phrase_row_budgets = atomic_phrase_validity_row_budgets(data)
    row_stability_radii = stability_cell_budgets.min(axis=1)
    row_validity_weak_radii = validity_cell_budgets.min(axis=1)
    targeted_sequence_baselines = aggregate_tpa_sequence_baselines(targeted_validity_cell_budgets)
    independent_stability_row_costs = stability_cell_budgets.sum(axis=1)
    independent_validity_row_costs = validity_cell_budgets.sum(axis=1)
    return {
        "raw_dpa_stab_min_cell": int(np.min(phd_margin_stability_budgets(data))),
        "dpa_stab_cell_min": int(np.min(stability_cell_budgets)),
        "dpa_stab_row_radius_q1": int(np.min(row_stability_radii)),
        "dpa_stab_row_radius_qN": int(np.max(row_stability_radii)),
        "dpa_val_cell_min": int(np.min(validity_cell_budgets)),
        "dpa_val_row_weak_q1": int(np.min(row_validity_weak_radii)),
        "dpa_val_row_weak_qN": int(np.max(row_validity_weak_radii)),
        "raw_dpa_val_min_cell": int(np.min(validity_cell_budgets)),
        "tpa_val_cell_min": int(np.min(targeted_validity_cell_budgets)),
        **targeted_sequence_baselines,
        "independent_stab_full_row_q1": int(np.min(independent_stability_row_costs)),
        "independent_stab_full_row_qN": int(independent_stability_row_costs.sum()),
        "independent_stab_qN_rL": int(independent_stability_row_costs.sum()),
        "independent_val_sequence_q1": int(np.min(independent_validity_row_costs)),
        "independent_val_sequence_qN": int(independent_validity_row_costs.sum()),
        "independent_val_q1": int(np.min(independent_validity_row_costs)),
        "independent_val_qN": int(independent_validity_row_costs.sum()),
        "phrase_dpa_val_q1": int(np.min(phrase_row_budgets)),
        "phrase_dpa_val_qN": int(np.max(phrase_row_budgets)),
        "phrase_independent_val_q1": int(np.min(phrase_row_budgets)),
        "phrase_independent_val_qN": int(phrase_row_budgets.sum()),
    }


def cell_stability_budgets(data: ToyData) -> np.ndarray:
    """Compute independent token-level stability budgets for baselines."""
    K, N, L = data.stab_votes.shape
    T = data.stab_counts.shape[2]
    budgets = np.zeros((N, L), dtype=np.int64)
    for i in range(N):
        for j in range(L):
            w = int(data.clean_pred[i, j])
            competitor_budgets = []
            for c in range(T):
                if c == w:
                    continue
                deficit = int(data.stab_counts[i, j, w] - data.stab_counts[i, j, c])
                contributions = [
                    int(data.influence[k, i, j]) * (int(data.stab_votes[k, i, j] != c) + int(data.stab_votes[k, i, j] == w))
                    for k in range(K)
                ]
                competitor_budgets.append(min_budget_from_contributions(deficit, contributions))
            budgets[i, j] = min(competitor_budgets)
    return budgets


def cell_validity_budgets(data: ToyData) -> np.ndarray:
    """Compute independent token-level harmful-target validity budgets."""
    K, N, L = data.val_votes.shape
    T = data.val_counts.shape[2]
    budgets = np.zeros((N, L), dtype=np.int64)
    for i in range(N):
        for j in range(L):
            h = int(data.target[i, j])
            target_count = int(data.val_counts[i, j, h])
            deficits = np.array([int(data.val_counts[i, j, c]) - target_count if c != h else 0 for c in range(T)], dtype=np.int64)
            contribs = np.zeros((K, T), dtype=np.int64)
            for k in range(K):
                if not int(data.influence[k, i, j]):
                    continue
                vote = int(data.val_votes[k, i, j])
                add_target = int(vote != h)
                for c in range(T):
                    if c != h:
                        contribs[k, c] = add_target + int(vote == c)
            budgets[i, j] = min_budget_satisfying_all(deficits, contribs, ignored_class=h)
    return budgets


def targeted_partition_radius(counts: np.ndarray, target: int, *, tie_wins: bool = True) -> int:
    """Return the TPA-style targeted token validity radius for one count vector."""
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1:
        raise ValueError("counts must be a one-dimensional token count vector")
    if not 0 <= target < counts.shape[0]:
        raise ValueError("target must index counts")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")

    target_count = int(counts[target])
    competitor_counts = np.delete(counts, target).astype(np.int64)
    if competitor_counts.size == 0:
        return 0

    already_succeeds = target_count >= int(np.max(competitor_counts)) if tie_wins else target_count > int(np.max(competitor_counts))
    if already_succeeds:
        return 0

    total_non_target_votes = int(np.sum(competitor_counts))
    for budget in range(total_non_target_votes + 1):
        target_after = target_count + budget
        max_competitor_after = target_after if tie_wins else target_after - 1
        required_removals = int(np.maximum(0, competitor_counts - max_competitor_after).sum())
        if required_removals <= budget:
            return budget
    return total_non_target_votes


def aggregate_tpa_sequence_baselines(token_radii: np.ndarray) -> dict[str, int | float]:
    """Aggregate token-level targeted radii into sequence-level TPA baselines."""
    token_radii = np.asarray(token_radii, dtype=np.int64)
    if token_radii.ndim != 2:
        raise ValueError("token_radii must have shape (N, L)")
    if token_radii.size == 0:
        raise ValueError("token_radii must be non-empty")
    row_sequence_radii = token_radii.max(axis=1)
    return {
        "tpa_val_sequence_q1": int(np.min(row_sequence_radii)),
        "tpa_val_sequence_qN": int(np.max(row_sequence_radii)),
        "tpa_val_sequence_mean": float(np.mean(row_sequence_radii)),
    }


def targeted_validity_token_budgets(data: ToyData) -> np.ndarray:
    """Compute per-cell TPA-style targeted harmful-token validity radii."""
    N, L, _ = data.val_counts.shape
    budgets = np.zeros((N, L), dtype=np.int64)
    for i in range(N):
        for j in range(L):
            budgets[i, j] = targeted_partition_radius(data.val_counts[i, j], int(data.target[i, j]))
    return budgets


def phd_margin_stability_budgets(data: ToyData) -> np.ndarray:
    margins = stability_margins(data.stab_counts, data.clean_pred, data.runner_up)
    return ((margins + 1) // 2).astype(np.int64)


def atomic_phrase_validity_row_budgets(data: ToyData) -> np.ndarray:
    """Compute atomic full-sequence budgets treating each generated sequence as one label."""
    K, N, _ = data.val_votes.shape
    budgets = np.zeros(N, dtype=np.int64)
    for i in range(N):
        target_phrase = tuple(int(x) for x in data.target[i])
        phrases = [tuple(int(x) for x in data.val_votes[k, i]) for k in range(K)]
        target_count = sum(phrase == target_phrase for phrase in phrases)
        competitor_counts: dict[tuple[int, ...], int] = {}
        for phrase in phrases:
            if phrase == target_phrase:
                continue
            competitor_counts[phrase] = competitor_counts.get(phrase, 0) + 1
        if not competitor_counts:
            budgets[i] = 0
            continue
        competitor_phrases = list(competitor_counts)
        deficits = np.array([competitor_counts[phrase] - target_count for phrase in competitor_phrases], dtype=np.int64)
        contribs = np.zeros((K, len(competitor_phrases)), dtype=np.int64)
        for k, phrase in enumerate(phrases):
            add_target = int(phrase != target_phrase)
            for c_idx, competitor_phrase in enumerate(competitor_phrases):
                contribs[k, c_idx] = add_target + int(phrase == competitor_phrase)
        budgets[i] = min_budget_satisfying_all(deficits, contribs)
    return budgets


def min_budget_from_contributions(deficit: int, contributions: list[int]) -> int:
    if deficit <= 0:
        return 0
    running = 0
    for budget, contribution in enumerate(sorted(contributions, reverse=True), start=1):
        running += contribution
        if running >= deficit:
            return budget
    return len(contributions) + 1


def min_budget_satisfying_all(deficits: np.ndarray, contribs: np.ndarray, ignored_class: int | None = None) -> int:
    active_deficits = deficits.copy()
    if ignored_class is not None:
        active_deficits[ignored_class] = 0
    if np.all(active_deficits <= 0):
        return 0

    useful = np.flatnonzero(np.any(contribs > 0, axis=1))
    no_solution_budget = int(contribs.shape[0] + 1)
    if useful.size == 0:
        return no_solution_budget

    useful_contribs = contribs[useful]
    useful_contribs = useful_contribs[np.argsort(-useful_contribs.sum(axis=1))]

    greedy_covered = np.zeros_like(active_deficits)
    remaining = list(range(useful_contribs.shape[0]))
    greedy_chosen = 0
    while remaining and not np.all(greedy_covered >= active_deficits):
        best_idx = max(remaining, key=lambda idx: int(np.minimum(useful_contribs[idx], np.maximum(0, active_deficits - greedy_covered)).sum()))
        greedy_covered += useful_contribs[best_idx]
        remaining.remove(best_idx)
        greedy_chosen += 1
    best = greedy_chosen if np.all(greedy_covered >= active_deficits) else no_solution_budget

    suffix_capacity = np.zeros((useful_contribs.shape[0] + 1, active_deficits.shape[0]), dtype=np.int64)
    for idx in range(useful_contribs.shape[0] - 1, -1, -1):
        suffix_capacity[idx] = suffix_capacity[idx + 1] + useful_contribs[idx]

    def search(idx: int, chosen: int, covered: np.ndarray) -> None:
        nonlocal best
        if np.all(covered >= active_deficits):
            best = min(best, chosen)
            return
        if idx == useful_contribs.shape[0] or chosen >= best:
            return
        if np.any(covered + suffix_capacity[idx] < active_deficits):
            return

        search(idx + 1, chosen + 1, covered + useful_contribs[idx])
        search(idx + 1, chosen, covered)

    search(0, 0, np.zeros_like(active_deficits))
    return int(best)
