#!/usr/bin/env python3
"""Fixed-budget certification curves for VPA vote-vector JSONL outputs.

Inherited DPA/TPA baselines use final tool-call vote counts. The proposed joint
MILPs use the shard-aware prompt-token grid. Token-grid DPA curves are optional
diagnostics rather than main full-scale baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toy_experiments.baselines import targeted_partition_radius


OBJECTIVE_MODE = "fixed_budget_adversarial_success"
EXPECTED_NUM_SHARDS = 500


@dataclass(frozen=True)
class PromptRow:
    original_row_index: int
    majority_class: str
    vote_vector: tuple[str, ...]
    token_vote_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class LoadReport:
    total_rows: int
    retained_rows: int
    filtered_short_rows: int
    truncated_rows: int
    stopped_at_max_prompts: bool


@dataclass(frozen=True)
class ValidityTarget:
    prompt_index: int
    target_class: str
    representative_shard_index: int
    active_positions: tuple[int, ...]
    target_tokens: tuple[int, ...]


@dataclass(frozen=True)
class StabilityEvent:
    prompt_index: int
    position: int
    competitor: int
    margin: int
    damage: tuple[int, ...]


@dataclass(frozen=True)
class ValidityEvent:
    target_index: int
    prompt_index: int
    position: int
    target_token: int
    competitor_margins_and_damage: tuple[tuple[int, tuple[int, ...]], ...]


def sorted_counter_items(counter: Counter[Any]) -> list[tuple[Any, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))


def majority_value(values: Iterable[Any]) -> Any:
    counts = Counter(values)
    if not counts:
        raise ValueError("Cannot compute a majority from an empty collection")
    return sorted_counter_items(counts)[0][0]


def dpa_certified_radius(winner_votes: int, competitor_votes: int) -> int:
    return max(0, (winner_votes - competitor_votes - 1) // 2)


def load_prompt_rows(
    path: Path,
    horizon: int,
    max_prompts: int | None,
    expected_num_shards: int = EXPECTED_NUM_SHARDS,
) -> tuple[list[PromptRow], LoadReport]:
    if horizon < 1:
        raise ValueError("--horizon must be positive")
    if max_prompts is not None and max_prompts < 1:
        raise ValueError("--max-prompts must be positive")

    rows: list[PromptRow] = []
    total_rows = 0
    filtered_short_rows = 0
    truncated_rows = 0
    stopped_at_max_prompts = False

    with path.open() as handle:
        for line_index, line in enumerate(handle):
            total_rows += 1
            raw = json.loads(line)
            vote_vector = raw["vote_vector"]
            matrix = raw["token_vote_matrix"]

            if len(vote_vector) != expected_num_shards or len(matrix) != expected_num_shards:
                raise ValueError(
                    f"Row {line_index}: expected len(vote_vector) == "
                    f"len(token_vote_matrix) == {expected_num_shards}, got "
                    f"{len(vote_vector)} and {len(matrix)}"
                )
            lengths = {len(sequence) for sequence in matrix}
            if len(lengths) != 1:
                raise ValueError(
                    f"Row {line_index}: shard token sequence lengths differ: {sorted(lengths)}"
                )
            non_none_prefix_lengths = [
                next(
                    (
                        index
                        for index, token in enumerate(sequence)
                        if token is None
                    ),
                    len(sequence),
                )
                for sequence in matrix
            ]
            usable_sequence_length = min(non_none_prefix_lengths)
            if usable_sequence_length < horizon:
                filtered_short_rows += 1
                continue

            for shard_index, sequence in enumerate(matrix):
                prefix = sequence[:horizon]
                if any(token is None for token in prefix):
                    raise ValueError(
                        f"Row {line_index}, shard {shard_index}: "
                        "None found inside retained horizon"
                    )

            actual_counts = Counter(vote_vector)
            stored_counts = Counter(raw["vote_counts"])
            if stored_counts != actual_counts:
                raise ValueError(
                    f"Row {line_index}: vote_counts does not equal Counter(vote_vector)"
                )
            stored_majority = raw["majority"]
            computed_majority = majority_value(vote_vector)
            if stored_majority != computed_majority:
                raise ValueError(
                    f"Row {line_index}: majority={stored_majority!r} does not match "
                    f"the deterministic vote_vector majority {computed_majority!r}"
                )

            if usable_sequence_length > horizon:
                truncated_rows += 1
            rows.append(
                PromptRow(
                    original_row_index=line_index,
                    majority_class=stored_majority,
                    vote_vector=tuple(str(value) for value in vote_vector),
                    token_vote_matrix=tuple(
                        tuple(int(token) for token in sequence[:horizon])
                        for sequence in matrix
                    ),
                )
            )
            if max_prompts is not None and len(rows) >= max_prompts:
                stopped_at_max_prompts = True
                break

    report = LoadReport(
        total_rows=total_rows,
        retained_rows=len(rows),
        filtered_short_rows=filtered_short_rows,
        truncated_rows=truncated_rows,
        stopped_at_max_prompts=stopped_at_max_prompts,
    )
    return rows, report


def build_grid(rows: list[PromptRow], horizon: int) -> np.ndarray:
    if not rows:
        raise ValueError(f"No prompts retained for horizon {horizon}")
    grids = []
    for row in rows:
        matrix = np.asarray(row.token_vote_matrix, dtype=np.int64)
        if matrix.shape != (EXPECTED_NUM_SHARDS, horizon):
            raise ValueError(
                f"Row {row.original_row_index}: expected token matrix shape "
                f"{(EXPECTED_NUM_SHARDS, horizon)}, got {matrix.shape}"
            )
        grids.append(matrix.T)
    return np.stack(grids, axis=0)


def clean_token_grid(grid: np.ndarray) -> np.ndarray:
    n, horizon, _ = grid.shape
    clean = np.empty((n, horizon), dtype=np.int64)
    for prompt_index in range(n):
        for position in range(horizon):
            clean[prompt_index, position] = int(
                majority_value(int(token) for token in grid[prompt_index, position])
            )
    return clean


def observed_target_classes(row: PromptRow) -> list[str]:
    counts = Counter(row.vote_vector)
    return [
        str(target)
        for target, _ in sorted_counter_items(counts)
        if target != row.majority_class
    ]


def build_validity_targets(
    rows: list[PromptRow],
    grid: np.ndarray,
    max_targets_per_prompt: int | None,
) -> list[ValidityTarget]:
    if max_targets_per_prompt is not None and max_targets_per_prompt < 1:
        raise ValueError("--max-targets-per-prompt must be positive")
    clean = clean_token_grid(grid)
    horizon = grid.shape[1]
    targets: list[ValidityTarget] = []

    for prompt_index, row in enumerate(rows):
        target_classes = observed_target_classes(row)
        if max_targets_per_prompt is not None:
            target_classes = target_classes[:max_targets_per_prompt]
        for target_class in target_classes:
            representative = next(
                shard
                for shard, prediction in enumerate(row.vote_vector)
                if prediction == target_class
            )
            target_tokens = row.token_vote_matrix[representative][:horizon]
            active_positions = tuple(
                position
                for position, target_token in enumerate(target_tokens)
                if target_token != int(clean[prompt_index, position])
            )
            if not active_positions:
                continue
            targets.append(
                ValidityTarget(
                    prompt_index=prompt_index,
                    target_class=target_class,
                    representative_shard_index=representative,
                    active_positions=active_positions,
                    target_tokens=target_tokens,
                )
            )
    return targets


def compute_dpa_final_tool_stability_radii(
    rows: list[PromptRow],
) -> list[float]:
    """Compute DPA stability from final tool-call vote counts only."""
    radii = []
    for row in rows:
        ranked = sorted_counter_items(Counter(row.vote_vector))
        winner_votes = ranked[0][1]
        runner_up_votes = ranked[1][1] if len(ranked) > 1 else 0
        radii.append(float(dpa_certified_radius(winner_votes, runner_up_votes)))
    return radii


def compute_dpa_token_grid_stability_radii(grid: np.ndarray) -> list[int]:
    """Compute the optional weakest-token DPA diagnostic on the token grid."""
    radii = []
    for prompt in grid:
        token_radii = []
        for shard_votes in prompt:
            ranked = sorted_counter_items(Counter(int(token) for token in shard_votes))
            winner_votes = ranked[0][1]
            runner_up_votes = ranked[1][1] if len(ranked) > 1 else 0
            token_radii.append(dpa_certified_radius(winner_votes, runner_up_votes))
        radii.append(min(token_radii))
    return radii


def compute_aggregate_tpa_final_tool_validity_radii(
    rows: list[PromptRow],
) -> list[float]:
    """Compute aggregate TPA validity from final tool-call vote counts only.

    This inherited baseline applies targeted partition aggregation to
    ``vote_vector`` counts. It does not use the token grid, shard identities,
    an MILP, or a shared poisoned-shard allocation.
    """
    radii: list[float] = []
    for row in rows:
        counts = Counter(row.vote_vector)
        targets = observed_target_classes(row)
        if not targets:
            radii.append(math.inf)
            continue
        classes = [name for name, _ in sorted(counts.items(), key=lambda item: str(item[0]))]
        count_vector = np.asarray([counts[name] for name in classes], dtype=np.int64)
        class_to_index = {name: index for index, name in enumerate(classes)}
        minimum_attack_budgets = [
            targeted_partition_radius(
                count_vector,
                class_to_index[target],
                tie_wins=True,
            )
            for target in targets
        ]
        radii.append(float(max(0, min(minimum_attack_budgets) - 1)))
    return radii


def compute_dpa_target_radii(
    rows: list[PromptRow],
    grid: np.ndarray,
    targets: list[ValidityTarget],
) -> list[float]:
    target_radii_by_prompt: dict[int, list[int]] = defaultdict(list)
    for target in targets:
        token_radii = []
        for position in target.active_positions:
            counts = Counter(
                int(token) for token in grid[target.prompt_index, position]
            )
            target_token = int(target.target_tokens[position])
            target_votes = counts.get(target_token, 0)
            strongest_non_target = max(
                (votes for token, votes in counts.items() if token != target_token),
                default=0,
            )
            token_radii.append(
                dpa_certified_radius(strongest_non_target, target_votes)
            )
        target_radii_by_prompt[target.prompt_index].append(max(token_radii))
    return [
        float(min(target_radii_by_prompt[index]))
        if target_radii_by_prompt[index]
        else math.inf
        for index in range(len(rows))
    ]


def baseline_curve_rows(
    method: str,
    radii: list[float],
    budgets: list[int],
) -> list[dict[str, Any]]:
    return [
        {
            "budget": budget,
            "method": method,
            "num_prompts": len(radii),
            "certified_prompts": sum(radius >= budget for radius in radii),
            "certified_fraction": sum(radius >= budget for radius in radii)
            / len(radii),
        }
        for budget in budgets
    ]


def build_stability_events(
    grid: np.ndarray,
    top_competitors: int,
) -> list[StabilityEvent]:
    if top_competitors < 1:
        raise ValueError("--top-competitors must be positive")
    events: list[StabilityEvent] = []
    n, horizon, num_shards = grid.shape
    for prompt_index in range(n):
        for position in range(horizon):
            votes = grid[prompt_index, position]
            counts = Counter(int(token) for token in votes)
            ranked = sorted_counter_items(counts)
            clean_token, clean_votes = int(ranked[0][0]), int(ranked[0][1])
            competitors = [
                int(token)
                for token, _ in ranked
                if int(token) != clean_token
            ][:top_competitors]
            for competitor in competitors:
                damage = tuple(
                    int(int(votes[shard]) == clean_token)
                    + int(int(votes[shard]) != competitor)
                    for shard in range(num_shards)
                )
                events.append(
                    StabilityEvent(
                        prompt_index=prompt_index,
                        position=position,
                        competitor=competitor,
                        margin=clean_votes - counts[competitor],
                        damage=damage,
                    )
                )
    return events


def build_validity_events(
    grid: np.ndarray,
    targets: list[ValidityTarget],
) -> list[ValidityEvent]:
    events: list[ValidityEvent] = []
    num_shards = grid.shape[2]
    for target_index, target in enumerate(targets):
        for position in target.active_positions:
            votes = grid[target.prompt_index, position]
            counts = Counter(int(token) for token in votes)
            target_token = int(target.target_tokens[position])
            target_votes = counts.get(target_token, 0)
            constraints = []
            for competitor, competitor_votes in sorted_counter_items(counts):
                competitor = int(competitor)
                if competitor == target_token:
                    continue
                damage = tuple(
                    int(int(votes[shard]) == competitor)
                    + int(int(votes[shard]) != target_token)
                    for shard in range(num_shards)
                )
                constraints.append((int(competitor_votes - target_votes), damage))
            events.append(
                ValidityEvent(
                    target_index=target_index,
                    prompt_index=target.prompt_index,
                    position=position,
                    target_token=target_token,
                    competitor_margins_and_damage=tuple(constraints),
                )
            )
    return events


def require_gurobi() -> tuple[Any, Any]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:
        raise RuntimeError(
            "Gurobi was selected but gurobipy is unavailable. Check the installation "
            "and license with:\n"
            "python - <<'PY'\n"
            "import gurobipy as gp\n"
            "print(gp.gurobi.version())\n"
            "m = gp.Model()\n"
            'print("Gurobi OK")\n'
            "PY"
        ) from exc
    return gp, GRB


def configure_gurobi_model(
    name: str,
    time_limit: float | None,
    mip_gap: float | None,
    threads: int,
    quiet: bool,
) -> tuple[Any, Any, Any]:
    gp, GRB = require_gurobi()
    model = gp.Model(name)
    model.Params.OutputFlag = 0 if quiet else 1
    model.Params.Threads = threads
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap
    return model, gp, GRB


def safe_model_float(model: Any, attribute: str) -> float | None:
    try:
        value = float(getattr(model, attribute))
    except Exception:
        return None
    return value if math.isfinite(value) else None


def gurobi_status_name(status: int, GRB: Any) -> str:
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return names.get(status, f"STATUS_{status}")


def conservative_failed_bound(
    num_prompts: int,
    objective_bound: float | None,
) -> int | None:
    if objective_bound is None:
        return None
    return min(num_prompts, max(0, int(math.ceil(objective_bound - 1e-9))))


def common_milp_row(
    *,
    budget: int,
    method: str,
    n: int,
    horizon: int,
    num_shards: int,
    num_events: int,
    solver_status: str,
    objective_value: float | None,
    objective_bound: float | None,
    mip_gap: float | None,
    elapsed: float,
    top_competitors: int,
    max_targets_per_prompt: int | None,
) -> dict[str, Any]:
    failed_bound = conservative_failed_bound(n, objective_bound)
    return {
        "budget": budget,
        "method": method,
        "objective_mode": OBJECTIVE_MODE,
        "num_prompts": n,
        "horizon": horizon,
        "num_shards": num_shards,
        "num_events": num_events,
        "max_failed_prompts": failed_bound,
        "certified_prompts_lower_bound": (
            None if failed_bound is None else n - failed_bound
        ),
        "certified_fraction_lower_bound": (
            None if failed_bound is None else (n - failed_bound) / n
        ),
        "solver": "gurobi",
        "solver_status": solver_status,
        "objective_value": objective_value,
        "objective_bound": objective_bound,
        "mip_gap": mip_gap,
        "time_seconds": elapsed,
        "top_competitors": top_competitors,
        "max_targets_per_prompt": max_targets_per_prompt,
    }


def solve_stability_gurobi(
    grid: np.ndarray,
    events: list[StabilityEvent],
    budget: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    n, horizon, num_shards = grid.shape
    model, gp, GRB = configure_gurobi_model(
        "fixed_budget_joint_row_column_stability",
        args.milp_time_limit,
        args.mip_gap,
        args.threads,
        args.quiet_gurobi,
    )
    a = model.addVars(num_shards, vtype=GRB.BINARY, name="a")
    y = model.addVars(len(events), vtype=GRB.BINARY, name="y")
    p = model.addVars(n, vtype=GRB.BINARY, name="p")
    model.addConstr(gp.quicksum(a.values()) <= budget, name="poison_budget")
    events_by_prompt: dict[int, list[int]] = defaultdict(list)
    for event_index, event in enumerate(events):
        model.addConstr(
            event.margin * y[event_index]
            <= gp.quicksum(
                event.damage[shard] * a[shard]
                for shard in range(num_shards)
                if event.damage[shard]
            ),
            name=f"event_{event_index}",
        )
        model.addConstr(y[event_index] <= p[event.prompt_index])
        events_by_prompt[event.prompt_index].append(event_index)
    for prompt_index in range(n):
        model.addConstr(
            p[prompt_index]
            <= gp.quicksum(y[index] for index in events_by_prompt[prompt_index])
        )
    model.setObjective(gp.quicksum(p.values()), GRB.MAXIMIZE)
    start = time.monotonic()
    model.optimize()
    elapsed = time.monotonic() - start
    objective_value = safe_model_float(model, "ObjVal") if model.SolCount else None
    return common_milp_row(
        budget=budget,
        method="joint_row_column_stability_milp",
        n=n,
        horizon=horizon,
        num_shards=num_shards,
        num_events=len(events),
        solver_status=gurobi_status_name(model.Status, GRB),
        objective_value=objective_value,
        objective_bound=safe_model_float(model, "ObjBound"),
        mip_gap=safe_model_float(model, "MIPGap") if model.SolCount else None,
        elapsed=elapsed,
        top_competitors=args.top_competitors,
        max_targets_per_prompt=args.max_targets_per_prompt,
    )


def solve_validity_gurobi(
    grid: np.ndarray,
    targets: list[ValidityTarget],
    events: list[ValidityEvent],
    budget: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    n, horizon, num_shards = grid.shape
    model, gp, GRB = configure_gurobi_model(
        "fixed_budget_joint_row_column_validity",
        args.milp_time_limit,
        args.mip_gap,
        args.threads,
        args.quiet_gurobi,
    )
    a = model.addVars(num_shards, vtype=GRB.BINARY, name="a")
    z = model.addVars(len(events), vtype=GRB.BINARY, name="z")
    g = model.addVars(len(targets), vtype=GRB.BINARY, name="g")
    p = model.addVars(n, vtype=GRB.BINARY, name="p")
    model.addConstr(gp.quicksum(a.values()) <= budget, name="poison_budget")
    events_by_target: dict[int, list[int]] = defaultdict(list)
    targets_by_prompt: dict[int, list[int]] = defaultdict(list)
    for event_index, event in enumerate(events):
        for competitor_index, (margin, damage) in enumerate(
            event.competitor_margins_and_damage
        ):
            model.addConstr(
                margin * z[event_index]
                <= gp.quicksum(
                    damage[shard] * a[shard]
                    for shard in range(num_shards)
                    if damage[shard]
                ),
                name=f"event_{event_index}_competitor_{competitor_index}",
            )
        events_by_target[event.target_index].append(event_index)
    for target_index, target in enumerate(targets):
        targets_by_prompt[target.prompt_index].append(target_index)
        for event_index in events_by_target[target_index]:
            model.addConstr(g[target_index] <= z[event_index])
        model.addConstr(g[target_index] <= p[target.prompt_index])
    for prompt_index in range(n):
        model.addConstr(
            p[prompt_index]
            <= gp.quicksum(g[index] for index in targets_by_prompt[prompt_index])
        )
    model.setObjective(gp.quicksum(p.values()), GRB.MAXIMIZE)
    start = time.monotonic()
    model.optimize()
    elapsed = time.monotonic() - start
    objective_value = safe_model_float(model, "ObjVal") if model.SolCount else None
    return common_milp_row(
        budget=budget,
        method="joint_row_column_validity_milp",
        n=n,
        horizon=horizon,
        num_shards=num_shards,
        num_events=len(events),
        solver_status=gurobi_status_name(model.Status, GRB),
        objective_value=objective_value,
        objective_bound=safe_model_float(model, "ObjBound"),
        mip_gap=safe_model_float(model, "MIPGap") if model.SolCount else None,
        elapsed=elapsed,
        top_competitors=args.top_competitors,
        max_targets_per_prompt=args.max_targets_per_prompt,
    )


def parse_budgets(raw: str) -> list[int]:
    budgets = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not budgets or budgets[0] < 0:
        raise ValueError("--budgets must contain non-negative integers")
    return budgets


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summary_curve_rows(
    baseline_groups: list[list[dict[str, Any]]],
    milp_groups: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in baseline_groups:
        for row in group:
            rows.append(
                {
                    "budget": row["budget"],
                    "method": row["method"],
                    "objective_mode": "radius_derived",
                    "num_prompts": row["num_prompts"],
                    "certified_prompts": row["certified_prompts"],
                    "certified_fraction": row["certified_fraction"],
                }
            )
    for group in milp_groups:
        rows.extend(group)
    return sorted(rows, key=lambda row: (int(row["budget"]), str(row["method"])))


def select_summary_baseline_groups(
    dpa_final_tool_stability: list[dict[str, Any]],
    aggregate_tpa_final_tool_validity: list[dict[str, Any]],
    dpa_token_grid_stability_diagnostic: list[dict[str, Any]],
    dpa_token_grid_validity_diagnostic: list[dict[str, Any]],
    *,
    include_token_grid_dpa_stability_diagnostic: bool,
    include_token_grid_dpa_validity_diagnostic: bool,
) -> list[list[dict[str, Any]]]:
    """Select main final-tool baselines plus explicitly requested diagnostics."""
    groups = [
        dpa_final_tool_stability,
        aggregate_tpa_final_tool_validity,
    ]
    if include_token_grid_dpa_stability_diagnostic:
        groups.append(dpa_token_grid_stability_diagnostic)
    if include_token_grid_dpa_validity_diagnostic:
        groups.append(dpa_token_grid_validity_diagnostic)
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute baseline and joint row-column fixed-budget adversarial "
            "success certification curves from existing VPA vote vectors."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--budgets", default="0,1,3,5,7,9,25,50,100,150,200,249"
    )
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--max-targets-per-prompt", type=int)
    parser.add_argument("--top-competitors", type=int, default=1)
    parser.add_argument("--milp-time-limit", type=float)
    parser.add_argument("--mip-gap", type=float)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--quiet-gurobi", action="store_true")
    parser.add_argument(
        "--include-token-grid-dpa-stability-diagnostic",
        action="store_true",
    )
    parser.add_argument(
        "--include-token-grid-dpa-validity-diagnostic",
        action="store_true",
    )
    parser.add_argument("--skip-stability-milp", action="store_true")
    parser.add_argument("--skip-validity-milp", action="store_true")
    args = parser.parse_args()
    if args.threads < 0:
        parser.error("--threads must be non-negative")
    if args.mip_gap is not None and args.mip_gap < 0:
        parser.error("--mip-gap must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    budgets = parse_budgets(args.budgets)
    rows, load_report = load_prompt_rows(
        args.input, args.horizon, args.max_prompts
    )
    grid = build_grid(rows, args.horizon)
    targets = build_validity_targets(rows, grid, args.max_targets_per_prompt)
    stability_events = build_stability_events(grid, args.top_competitors)
    validity_events = build_validity_events(grid, targets)

    out_dir = args.output_dir / args.name / f"H{args.horizon:03d}"
    if args.max_prompts is not None:
        out_dir = out_dir / f"N{args.max_prompts:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dpa_final_tool_stability = baseline_curve_rows(
        "dpa_final_tool_stability",
        compute_dpa_final_tool_stability_radii(rows),
        budgets,
    )
    aggregate_tpa_final_tool_validity = baseline_curve_rows(
        "aggregate_tpa_final_tool_validity",
        compute_aggregate_tpa_final_tool_validity_radii(rows),
        budgets,
    )
    dpa_token_grid_stability_diagnostic: list[dict[str, Any]] = []
    if args.include_token_grid_dpa_stability_diagnostic:
        dpa_token_grid_stability_diagnostic = baseline_curve_rows(
            "dpa_token_grid_weakest_token_stability_diagnostic",
            compute_dpa_token_grid_stability_radii(grid),
            budgets,
        )
    dpa_token_grid_validity_diagnostic: list[dict[str, Any]] = []
    num_baseline_validity_targets = 0
    if args.include_token_grid_dpa_validity_diagnostic:
        baseline_targets = build_validity_targets(rows, grid, None)
        num_baseline_validity_targets = len(baseline_targets)
        dpa_token_grid_validity_diagnostic = baseline_curve_rows(
            "dpa_max_target_token_validity_diagnostic",
            compute_dpa_target_radii(rows, grid, baseline_targets),
            budgets,
        )
    write_rows(
        out_dir / "dpa_final_tool_stability.csv",
        dpa_final_tool_stability,
    )
    write_rows(
        out_dir / "aggregate_tpa_final_tool_validity.csv",
        aggregate_tpa_final_tool_validity,
    )
    if args.include_token_grid_dpa_stability_diagnostic:
        write_rows(
            out_dir / "dpa_token_grid_weakest_token_stability_diagnostic.csv",
            dpa_token_grid_stability_diagnostic,
        )
    if args.include_token_grid_dpa_validity_diagnostic:
        write_rows(
            out_dir / "dpa_max_target_token_validity_diagnostic.csv",
            dpa_token_grid_validity_diagnostic,
        )

    stability_rows: list[dict[str, Any]] = []
    validity_rows: list[dict[str, Any]] = []
    for budget in budgets:
        if not args.skip_stability_milp:
            print(f"[stability] solver=gurobi budget={budget}", flush=True)
            result = solve_stability_gurobi(
                grid, stability_events, budget, args
            )
            stability_rows.append(result)
            write_rows(
                out_dir / "joint_row_column_stability_milp.csv",
                stability_rows,
            )
        if not args.skip_validity_milp:
            print(f"[validity] solver=gurobi budget={budget}", flush=True)
            result = solve_validity_gurobi(
                grid, targets, validity_events, budget, args
            )
            validity_rows.append(result)
            write_rows(
                out_dir / "joint_row_column_validity_milp.csv",
                validity_rows,
            )

    baseline_groups = select_summary_baseline_groups(
        dpa_final_tool_stability,
        aggregate_tpa_final_tool_validity,
        dpa_token_grid_stability_diagnostic,
        dpa_token_grid_validity_diagnostic,
        include_token_grid_dpa_stability_diagnostic=(
            args.include_token_grid_dpa_stability_diagnostic
        ),
        include_token_grid_dpa_validity_diagnostic=(
            args.include_token_grid_dpa_validity_diagnostic
        ),
    )
    combined = summary_curve_rows(
        baseline_groups,
        [stability_rows, validity_rows],
    )
    write_rows(out_dir / "budget_curve_summary.csv", combined)
    summary = {
        "name": args.name,
        "input": str(args.input),
        "objective_mode": OBJECTIVE_MODE,
        "horizon": args.horizon,
        "num_shards": int(grid.shape[2]),
        "num_prompts": len(rows),
        "num_total_rows_read": load_report.total_rows,
        "num_rows_filtered_shorter_than_horizon": load_report.filtered_short_rows,
        "num_rows_truncated_to_horizon": load_report.truncated_rows,
        "horizon_filter_basis": "shortest_non_none_prefix_across_shards",
        "padding_policy": "no_padding_none_rows_filtered",
        "stopped_at_max_prompts": load_report.stopped_at_max_prompts,
        "budgets": budgets,
        "solver": "gurobi",
        "milp_time_limit": args.milp_time_limit,
        "mip_gap": args.mip_gap,
        "threads": args.threads,
        "top_competitors": args.top_competitors,
        "max_targets_per_prompt": args.max_targets_per_prompt,
        "num_stability_events": len(stability_events),
        "num_baseline_validity_targets": num_baseline_validity_targets,
        "num_validity_targets": len(targets),
        "num_validity_events": len(validity_events),
        "full_scale_baseline_interface": "final_tool_vote_vector",
        "proposed_method_interface": "shard_aware_prompt_token_grid",
        "main_stability_baseline": "dpa_final_tool_stability",
        "main_validity_baseline": "aggregate_tpa_final_tool_validity",
        "token_grid_dpa_stability_diagnostic_included": (
            args.include_token_grid_dpa_stability_diagnostic
        ),
        "token_grid_dpa_validity_diagnostic_included": (
            args.include_token_grid_dpa_validity_diagnostic
        ),
        "tpa_baseline": "aggregate_tpa_final_tool_validity",
        "tpa_baseline_uses_milp": False,
        "tpa_baseline_uses_shard_identities": False,
        "tpa_baseline_uses_shared_poisoning_allocation": False,
        "tie_convention": "adversarial_target_or_competitor_wins_ties",
        "validity_constraint": (
            "target_ties_or_beats_every_observed_non_target_token"
        ),
        "configurations_are_independent_clean_ensembles": True,
    }
    with (out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Wrote certification curves to {out_dir}")


if __name__ == "__main__":
    main()
