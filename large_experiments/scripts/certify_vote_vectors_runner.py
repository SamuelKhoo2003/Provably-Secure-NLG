#!/usr/bin/env python3
"""
Full-scale certification runner for VPA vote-vector JSONL outputs.

This runner intentionally skips row-only and column-only MILPs.

It computes:
- DPA weakest-token stability
- joint row-column stability MILP
- aggregate TPA MCP validity
- DPA max-target-token validity
- joint row-column validity MILP

Input:
- raw VPA vote-vector JSONL file, for example:
  /data/mwicker/VPA/vote_vectors_1b_full_gpu0.jsonl

The file is assumed to contain:
- vote_vector, length K
- token_vote_matrix, shape K x L_i
- vote_counts
- majority
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PromptRow:
    original_row_index: int
    majority_class: str
    vote_vector: list[str]
    token_vote_matrix: list[list[int]]
    ground_truth_correct: bool | None
    majority_is_safe: bool | None
    stored_robustness_radius: int | None


@dataclass(frozen=True)
class TokenCell:
    prompt_index: int
    position: int
    clean_token: int
    clean_votes: int
    counts: Counter[int]


@dataclass(frozen=True)
class ValidityTarget:
    prompt_index: int
    target_class: str
    target_class_votes: int
    representative_shard_index: int
    active_positions: tuple[int, ...]
    target_tokens: tuple[int, ...]


def dpa_radius_from_margin(winner_votes: int, competitor_votes: int) -> int:
    return max(0, (winner_votes - competitor_votes - 1) // 2)


def sorted_counter_items(counter: Counter[Any]) -> list[tuple[Any, int]]:
    return sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))


def load_prompt_rows(path: Path, horizon: int, max_prompts: int | None) -> tuple[list[PromptRow], int]:
    rows: list[PromptRow] = []
    total_rows = 0

    with path.open() as f:
        for line_idx, line in enumerate(f):
            total_rows += 1
            row = json.loads(line)

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
                    f"Row {line_idx}: inconsistent token lengths across shards: {lengths}"
                )

            seq_len = next(iter(lengths))
            if seq_len < horizon:
                continue

            stored_counts = dict(row.get("vote_counts", {}))
            actual_counts = dict(Counter(vote_vector))
            if stored_counts and stored_counts != actual_counts:
                raise ValueError(
                    f"Row {line_idx}: vote_counts does not match Counter(vote_vector)"
                )

            truncated_matrix = [seq[:horizon] for seq in matrix]

            rows.append(
                PromptRow(
                    original_row_index=line_idx,
                    majority_class=row["majority"],
                    vote_vector=vote_vector,
                    token_vote_matrix=truncated_matrix,
                    ground_truth_correct=row.get("ground_truth_correct"),
                    majority_is_safe=row.get("majority_is_safe"),
                    stored_robustness_radius=row.get("robustness_radius"),
                )
            )

            if max_prompts is not None and len(rows) >= max_prompts:
                break

    return rows, total_rows


def build_grid(rows: list[PromptRow], horizon: int) -> np.ndarray:
    """
    Return grid with shape N x H x K.

    row.token_vote_matrix has shape K x H.
    """
    prompt_grids = []
    for row in rows:
        mat = np.asarray(row.token_vote_matrix, dtype=np.int64)
        if mat.ndim != 2:
            raise ValueError("token_vote_matrix must be 2D")
        if mat.shape[1] != horizon:
            raise ValueError(f"Expected horizon {horizon}, got {mat.shape[1]}")
        prompt_grids.append(mat.T)

    return np.stack(prompt_grids, axis=0)


def compute_token_cells(grid: np.ndarray) -> list[TokenCell]:
    n, h, _k = grid.shape
    cells: list[TokenCell] = []

    for i in range(n):
        for pos in range(h):
            counts = Counter(int(x) for x in grid[i, pos, :])
            top = sorted_counter_items(counts)
            clean_token, clean_votes = top[0]
            cells.append(
                TokenCell(
                    prompt_index=i,
                    position=pos,
                    clean_token=int(clean_token),
                    clean_votes=int(clean_votes),
                    counts=counts,
                )
            )

    return cells


def compute_dpa_weakest_stability(
    rows: list[PromptRow],
    grid: np.ndarray,
    output_path: Path,
) -> list[int]:
    n, h, _k = grid.shape
    prompt_radii: list[int] = []

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_index",
                "original_row_index",
                "horizon",
                "weakest_dpa_stability_radius",
                "weakest_position",
                "clean_token_id",
                "clean_votes",
                "runner_up_token_id",
                "runner_up_votes",
            ],
        )
        writer.writeheader()

        for i in range(n):
            cell_infos = []

            for pos in range(h):
                counts = Counter(int(x) for x in grid[i, pos, :])
                top = sorted_counter_items(counts)
                clean_token, clean_votes = top[0]
                if len(top) > 1:
                    runner_token, runner_votes = top[1]
                else:
                    runner_token, runner_votes = "", 0

                radius = dpa_radius_from_margin(clean_votes, runner_votes)
                cell_infos.append(
                    (
                        radius,
                        pos,
                        clean_token,
                        clean_votes,
                        runner_token,
                        runner_votes,
                    )
                )

            weakest = min(cell_infos, key=lambda x: (x[0], x[1]))
            prompt_radii.append(int(weakest[0]))

            writer.writerow(
                {
                    "prompt_index": i,
                    "original_row_index": rows[i].original_row_index,
                    "horizon": h,
                    "weakest_dpa_stability_radius": weakest[0],
                    "weakest_position": weakest[1],
                    "clean_token_id": weakest[2],
                    "clean_votes": weakest[3],
                    "runner_up_token_id": weakest[4],
                    "runner_up_votes": weakest[5],
                }
            )

    return prompt_radii


def tpa_radius_from_counts(counts: Counter[str], target: str) -> int:
    """
    Exact small-K search for aggregate targeted plurality radius.

    Returns the largest certified budget r such that the target cannot be
    made a plurality winner using at most r changed shard votes.

    This uses the same conservative tie convention as DPA:
    if the target can tie the current winner after b changes, then b is
    treated as potentially unsafe, so the certified radius is b - 1.
    """
    k_total = sum(counts.values())
    target_votes = counts.get(target, 0)

    if target_votes >= max(counts.values()):
        return 0

    competitors = [v for cls, v in counts.items() if cls != target]

    for b in range(0, k_total - target_votes + 1):
        target_after = target_votes + b

        # Number of non-target votes that must be moved away from competitors
        # so that every competitor has at most target_after votes.
        required_removals = sum(max(0, v - target_after) for v in competitors)

        if required_removals <= b:
            return max(0, b - 1)

    return max(0, k_total - target_votes)


def compute_aggregate_tpa_mcp_validity(
    rows: list[PromptRow],
    output_path: Path,
) -> list[int]:
    prompt_radii: list[int] = []

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_index",
                "original_row_index",
                "majority_class",
                "target_class",
                "target_votes",
                "target_rank_by_votes",
                "tpa_radius",
                "prompt_aggregate_tpa_mcp_validity_radius",
            ],
        )
        writer.writeheader()

        for i, row in enumerate(rows):
            counts = Counter(row.vote_vector)
            targets = [
                cls for cls, _votes in sorted_counter_items(counts)
                if cls != row.majority_class
            ]

            target_results = []
            for rank, target in enumerate(targets, start=1):
                radius = tpa_radius_from_counts(counts, target)
                target_results.append((radius, target, counts[target], rank))

            if target_results:
                prompt_radius = min(r for r, _t, _v, _rank in target_results)
            else:
                prompt_radius = math.inf

            prompt_radii.append(0 if prompt_radius is math.inf else int(prompt_radius))

            for radius, target, votes, rank in target_results:
                writer.writerow(
                    {
                        "prompt_index": i,
                        "original_row_index": row.original_row_index,
                        "majority_class": row.majority_class,
                        "target_class": target,
                        "target_votes": votes,
                        "target_rank_by_votes": rank,
                        "tpa_radius": radius,
                        "prompt_aggregate_tpa_mcp_validity_radius": prompt_radius,
                    }
                )

    return prompt_radii


def build_validity_targets(
    rows: list[PromptRow],
    grid: np.ndarray,
    max_targets_per_prompt: int | None,
) -> list[ValidityTarget]:
    n, h, k = grid.shape
    targets: list[ValidityTarget] = []

    clean_tokens = np.zeros((n, h), dtype=np.int64)

    for i in range(n):
        for pos in range(h):
            counts = Counter(int(x) for x in grid[i, pos, :])
            clean_tokens[i, pos] = int(sorted_counter_items(counts)[0][0])

    for i, row in enumerate(rows):
        class_counts = Counter(row.vote_vector)
        target_classes = [
            cls for cls, _votes in sorted_counter_items(class_counts)
            if cls != row.majority_class
        ]

        if max_targets_per_prompt is not None:
            target_classes = target_classes[:max_targets_per_prompt]

        for target_class in target_classes:
            representative_shards = [
                shard_idx
                for shard_idx, pred in enumerate(row.vote_vector)
                if pred == target_class
            ]

            if not representative_shards:
                continue

            rep = representative_shards[0]
            target_tokens = tuple(int(x) for x in row.token_vote_matrix[rep][:h])

            active_positions = tuple(
                pos for pos in range(h)
                if target_tokens[pos] != int(clean_tokens[i, pos])
            )

            if not active_positions:
                continue

            targets.append(
                ValidityTarget(
                    prompt_index=i,
                    target_class=target_class,
                    target_class_votes=class_counts[target_class],
                    representative_shard_index=rep,
                    active_positions=active_positions,
                    target_tokens=target_tokens,
                )
            )

    return targets


def compute_dpa_max_target_token_validity(
    rows: list[PromptRow],
    grid: np.ndarray,
    targets: list[ValidityTarget],
    output_path: Path,
) -> list[int]:
    h = grid.shape[1]
    by_prompt: dict[int, list[int]] = defaultdict(list)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_index",
                "original_row_index",
                "target_class",
                "target_class_votes",
                "num_active_positions",
                "max_target_token_radius",
                "prompt_dpa_max_target_token_validity_radius",
            ],
        )
        writer.writeheader()

        temp_rows = []

        for target in targets:
            i = target.prompt_index
            token_radii = []

            for pos in target.active_positions:
                counts = Counter(int(x) for x in grid[i, pos, :])
                clean_token, clean_votes = sorted_counter_items(counts)[0]
                target_token = target.target_tokens[pos]
                target_votes = counts.get(target_token, 0)
                radius = dpa_radius_from_margin(clean_votes, target_votes)
                token_radii.append(radius)

            # Exact target sequence needs all active positions, so this is the
            # hardest active target token for that target class.
            target_radius = max(token_radii) if token_radii else math.inf
            by_prompt[i].append(int(target_radius))

            temp_rows.append(
                {
                    "prompt_index": i,
                    "original_row_index": rows[i].original_row_index,
                    "target_class": target.target_class,
                    "target_class_votes": target.target_class_votes,
                    "num_active_positions": len(target.active_positions),
                    "max_target_token_radius": target_radius,
                }
            )

        prompt_radius = {
            i: min(vals) for i, vals in by_prompt.items() if vals
        }

        for row in temp_rows:
            i = row["prompt_index"]
            row["prompt_dpa_max_target_token_validity_radius"] = prompt_radius.get(i, "")
            writer.writerow(row)

    return [min(by_prompt[i]) if by_prompt.get(i) else 0 for i in range(len(rows))]


def parse_budgets(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def require_scipy_milp():
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except Exception as exc:
        raise RuntimeError(
            "scipy.optimize.milp is required for the joint MILPs. "
            "Install a recent scipy in the large-experiments environment."
        ) from exc

    return Bounds, LinearConstraint, milp, lil_matrix


def select_stability_competitors(
    counts: Counter[int],
    clean_token: int,
    top_competitors: int,
) -> list[int]:
    competitors = [
        int(tok)
        for tok, _votes in sorted_counter_items(counts)
        if int(tok) != int(clean_token)
    ]
    return competitors[:top_competitors]


def solve_joint_stability_milp(
    grid: np.ndarray,
    budget: int,
    top_competitors: int,
    time_limit: float | None,
) -> dict[str, Any]:
    """
    Maximise the number of prompts whose horizon stability can be broken.

    A prompt is counted as broken if any token position can be changed to one
    of the selected competitor tokens.
    """
    Bounds, LinearConstraint, milp, lil_matrix = require_scipy_milp()

    n, h, k = grid.shape

    events = []
    for i in range(n):
        for pos in range(h):
            counts = Counter(int(x) for x in grid[i, pos, :])
            top = sorted_counter_items(counts)
            clean_token, clean_votes = int(top[0][0]), int(top[0][1])
            competitors = select_stability_competitors(
                counts,
                clean_token,
                top_competitors=top_competitors,
            )

            for comp in competitors:
                comp_votes = counts.get(comp, 0)
                margin = clean_votes - comp_votes
                if margin <= 0:
                    continue

                damage = np.zeros(k, dtype=np.int8)
                for shard in range(k):
                    tok = int(grid[i, pos, shard])
                    damage[shard] = int(tok == clean_token) + int(tok != comp)

                events.append((i, pos, comp, margin, damage))

    num_a = k
    num_y = len(events)
    num_p = n

    a_offset = 0
    y_offset = a_offset + num_a
    p_offset = y_offset + num_y
    num_vars = p_offset + num_p

    constraints = []
    lower = []
    upper = []

    # Poisoning budget.
    row = {}
    for shard in range(k):
        row[a_offset + shard] = 1.0
    constraints.append(row)
    lower.append(0.0)
    upper.append(float(budget))

    # Event feasibility and event to prompt linkage.
    events_by_prompt: dict[int, list[int]] = defaultdict(list)

    for e_idx, (i, _pos, _comp, margin, damage) in enumerate(events):
        y_var = y_offset + e_idx
        p_var = p_offset + i

        # margin * y - damage dot a <= 0
        row = {y_var: float(margin)}
        for shard in np.nonzero(damage)[0]:
            row[a_offset + int(shard)] = row.get(a_offset + int(shard), 0.0) - float(damage[shard])
        constraints.append(row)
        lower.append(-math.inf)
        upper.append(0.0)

        # y - p <= 0
        constraints.append({y_var: 1.0, p_var: -1.0})
        lower.append(-math.inf)
        upper.append(0.0)

        events_by_prompt[i].append(e_idx)

    # p_i <= sum events for prompt i
    for i in range(n):
        row = {p_offset + i: 1.0}
        for e_idx in events_by_prompt.get(i, []):
            row[y_offset + e_idx] = row.get(y_offset + e_idx, 0.0) - 1.0
        constraints.append(row)
        lower.append(-math.inf)
        upper.append(0.0)

    A = lil_matrix((len(constraints), num_vars), dtype=float)
    for r, coeffs in enumerate(constraints):
        for c, v in coeffs.items():
            A[r, c] = v

    c = np.zeros(num_vars)
    c[p_offset:p_offset + n] = -1.0

    integrality = np.ones(num_vars)
    bounds = Bounds(np.zeros(num_vars), np.ones(num_vars))
    linear_constraint = LinearConstraint(A.tocsr(), np.array(lower), np.array(upper))

    options = {}
    if time_limit is not None:
        options["time_limit"] = time_limit

    start = time.time()
    result = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=linear_constraint,
        options=options,
    )
    elapsed = time.time() - start

    if result.x is None:
        max_failed_prompts = None
    else:
        p_values = result.x[p_offset:p_offset + n]
        max_failed_prompts = int(round(float(np.sum(p_values))))

    return {
        "budget": budget,
        "num_prompts": n,
        "horizon": h,
        "num_shards": k,
        "num_events": len(events),
        "top_competitors": top_competitors,
        "max_failed_prompts": max_failed_prompts,
        "certified_prompts_lower_bound": None if max_failed_prompts is None else n - max_failed_prompts,
        "certified_fraction_lower_bound": None if max_failed_prompts is None else (n - max_failed_prompts) / n,
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "objective": None if result.fun is None else float(result.fun),
        "time_seconds": elapsed,
    }


def solve_joint_validity_milp(
    grid: np.ndarray,
    targets: list[ValidityTarget],
    budget: int,
    time_limit: float | None,
) -> dict[str, Any]:
    """
    Maximise the number of prompts for which any alternative tool-call target
    can be induced across all active target positions.
    """
    Bounds, LinearConstraint, milp, lil_matrix = require_scipy_milp()

    n, h, k = grid.shape

    # Expand target-position events.
    y_events = []
    target_to_y_indices: dict[int, list[int]] = defaultdict(list)

    for t_idx, target in enumerate(targets):
        i = target.prompt_index
        for pos in target.active_positions:
            counts = Counter(int(x) for x in grid[i, pos, :])
            clean_token, clean_votes = sorted_counter_items(counts)[0]
            clean_token = int(clean_token)
            clean_votes = int(clean_votes)
            target_token = int(target.target_tokens[pos])
            target_votes = int(counts.get(target_token, 0))

            margin = clean_votes - target_votes
            if margin <= 0:
                # Already at least tied under the conservative target criterion.
                margin = 0

            damage = np.zeros(k, dtype=np.int8)
            for shard in range(k):
                tok = int(grid[i, pos, shard])
                damage[shard] = int(tok == clean_token) + int(tok != target_token)

            y_events.append((t_idx, i, pos, margin, damage))
            target_to_y_indices[t_idx].append(len(y_events) - 1)

    num_a = k
    num_y = len(y_events)
    num_g = len(targets)
    num_p = n

    a_offset = 0
    y_offset = a_offset + num_a
    g_offset = y_offset + num_y
    p_offset = g_offset + num_g
    num_vars = p_offset + num_p

    constraints = []
    lower = []
    upper = []

    # Poisoning budget.
    row = {}
    for shard in range(k):
        row[a_offset + shard] = 1.0
    constraints.append(row)
    lower.append(0.0)
    upper.append(float(budget))

    # y event feasibility.
    for y_idx, (_t_idx, _i, _pos, margin, damage) in enumerate(y_events):
        y_var = y_offset + y_idx

        if margin <= 0:
            # y can be set without poisoning.
            continue

        row = {y_var: float(margin)}
        for shard in np.nonzero(damage)[0]:
            row[a_offset + int(shard)] = row.get(a_offset + int(shard), 0.0) - float(damage[shard])
        constraints.append(row)
        lower.append(-math.inf)
        upper.append(0.0)

    targets_by_prompt: dict[int, list[int]] = defaultdict(list)

    # g_t <= every y for that target.
    for t_idx, target in enumerate(targets):
        g_var = g_offset + t_idx
        targets_by_prompt[target.prompt_index].append(t_idx)

        for y_idx in target_to_y_indices[t_idx]:
            y_var = y_offset + y_idx
            constraints.append({g_var: 1.0, y_var: -1.0})
            lower.append(-math.inf)
            upper.append(0.0)

        # g_t implies p_i.
        p_var = p_offset + target.prompt_index
        constraints.append({g_var: 1.0, p_var: -1.0})
        lower.append(-math.inf)
        upper.append(0.0)

    # p_i <= sum target successes for prompt i.
    for i in range(n):
        row = {p_offset + i: 1.0}
        for t_idx in targets_by_prompt.get(i, []):
            row[g_offset + t_idx] = row.get(g_offset + t_idx, 0.0) - 1.0
        constraints.append(row)
        lower.append(-math.inf)
        upper.append(0.0)

    A = lil_matrix((len(constraints), num_vars), dtype=float)
    for r, coeffs in enumerate(constraints):
        for c_idx, v in coeffs.items():
            A[r, c_idx] = v

    c = np.zeros(num_vars)
    c[p_offset:p_offset + n] = -1.0

    integrality = np.ones(num_vars)
    bounds = Bounds(np.zeros(num_vars), np.ones(num_vars))
    linear_constraint = LinearConstraint(A.tocsr(), np.array(lower), np.array(upper))

    options = {}
    if time_limit is not None:
        options["time_limit"] = time_limit

    start = time.time()
    result = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=linear_constraint,
        options=options,
    )
    elapsed = time.time() - start

    if result.x is None:
        max_failed_prompts = None
    else:
        p_values = result.x[p_offset:p_offset + n]
        max_failed_prompts = int(round(float(np.sum(p_values))))

    return {
        "budget": budget,
        "num_prompts": n,
        "horizon": h,
        "num_shards": k,
        "num_targets": len(targets),
        "num_target_position_events": len(y_events),
        "max_failed_prompts": max_failed_prompts,
        "certified_prompts_lower_bound": None if max_failed_prompts is None else n - max_failed_prompts,
        "certified_fraction_lower_bound": None if max_failed_prompts is None else (n - max_failed_prompts) / n,
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "objective": None if result.fun is None else float(result.fun),
        "time_seconds": elapsed,
    }


def write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--budgets", default="0,1,3,5,7,9,25,50,100,150,200,249")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-targets-per-prompt", type=int, default=None)
    parser.add_argument("--top-competitors", type=int, default=1)
    parser.add_argument("--milp-time-limit", type=float, default=None)
    parser.add_argument("--skip-stability-milp", action="store_true")
    parser.add_argument("--skip-validity-milp", action="store_true")
    args = parser.parse_args()

    budgets = parse_budgets(args.budgets)

    rows, total_rows = load_prompt_rows(
        path=args.input,
        horizon=args.horizon,
        max_prompts=args.max_prompts,
    )

    if not rows:
        raise ValueError(f"No rows retained for horizon {args.horizon}")

    grid = build_grid(rows, args.horizon)
    targets = build_validity_targets(
        rows=rows,
        grid=grid,
        max_targets_per_prompt=args.max_targets_per_prompt,
    )

    out_dir = args.output_dir / args.name / f"H{args.horizon:03d}"
    if args.max_prompts is not None:
        out_dir = out_dir / f"N{args.max_prompts:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dpa_stability_radii = compute_dpa_weakest_stability(
        rows=rows,
        grid=grid,
        output_path=out_dir / "dpa_weakest_token_stability.csv",
    )

    aggregate_tpa_radii = compute_aggregate_tpa_mcp_validity(
        rows=rows,
        output_path=out_dir / "aggregate_tpa_mcp_validity.csv",
    )

    dpa_target_radii = compute_dpa_max_target_token_validity(
        rows=rows,
        grid=grid,
        targets=targets,
        output_path=out_dir / "dpa_max_target_token_validity.csv",
    )

    stability_milp_rows: list[dict[str, Any]] = []
    if not args.skip_stability_milp:
        for budget in budgets:
            print(f"[stability MILP] budget={budget}")
            stability_milp_rows.append(
                solve_joint_stability_milp(
                    grid=grid,
                    budget=budget,
                    top_competitors=args.top_competitors,
                    time_limit=args.milp_time_limit,
                )
            )
            write_dict_rows(out_dir / "joint_row_column_stability_milp.csv", stability_milp_rows)

    validity_milp_rows: list[dict[str, Any]] = []
    if not args.skip_validity_milp:
        for budget in budgets:
            print(f"[validity MILP] budget={budget}")
            validity_milp_rows.append(
                solve_joint_validity_milp(
                    grid=grid,
                    targets=targets,
                    budget=budget,
                    time_limit=args.milp_time_limit,
                )
            )
            write_dict_rows(out_dir / "joint_row_column_validity_milp.csv", validity_milp_rows)

    summary = {
        "name": args.name,
        "input": str(args.input),
        "horizon": args.horizon,
        "num_total_rows_in_file": total_rows,
        "num_retained_prompts": len(rows),
        "num_shards": int(grid.shape[2]),
        "num_validity_targets": len(targets),
        "budgets": budgets,
        "top_competitors": args.top_competitors,
        "max_targets_per_prompt": args.max_targets_per_prompt,
        "milp_time_limit": args.milp_time_limit,
        "dpa_weakest_stability_mean": float(np.mean(dpa_stability_radii)),
        "dpa_weakest_stability_median": float(np.median(dpa_stability_radii)),
        "dpa_weakest_stability_min": int(np.min(dpa_stability_radii)),
        "dpa_weakest_stability_max": int(np.max(dpa_stability_radii)),
        "aggregate_tpa_mcp_validity_mean": float(np.mean(aggregate_tpa_radii)),
        "aggregate_tpa_mcp_validity_median": float(np.median(aggregate_tpa_radii)),
        "aggregate_tpa_mcp_validity_min": int(np.min(aggregate_tpa_radii)),
        "aggregate_tpa_mcp_validity_max": int(np.max(aggregate_tpa_radii)),
        "dpa_max_target_token_validity_mean": float(np.mean(dpa_target_radii)),
        "dpa_max_target_token_validity_median": float(np.median(dpa_target_radii)),
        "dpa_max_target_token_validity_min": int(np.min(dpa_target_radii)),
        "dpa_max_target_token_validity_max": int(np.max(dpa_target_radii)),
    }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote outputs to {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()