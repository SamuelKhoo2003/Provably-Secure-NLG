"""Gurobi MILP solvers for shared row/column poisoning certificates.

The solvers in this module all optimize the same core quantity:
``B_star = min sum_k a[k]``, where ``a[k]`` is a binary poisoned-shard
allocation. The same allocation is reused across all required prompt rows and
token positions, which is the shared row-column MILP coupling.

Stability objectives change outputs away from the clean winner. Validity
objectives force harmful target tokens or full harmful target sequences. Report
figures should prefer :func:`solve_structured_stability` and
:func:`solve_row_col_validity`; older row/column functions remain as
compatibility wrappers and baseline-oriented objectives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gurobipy as gp
import numpy as np
from gurobipy import GRB


@dataclass(frozen=True)
class CertificateResult:
    """Result returned by a certificate MILP solve.

    ``B_star`` is the poisoned-shard budget. If ``is_optimal`` is false and a
    feasible solution exists, ``B_star`` is the best feasible upper bound found
    by Gurobi rather than a certified optimum. ``attacked_cells`` is diagnostic:
    attacked-cell indicators are not secondarily minimized, so the list may
    contain extra feasible cells beyond those required by the objective.
    """

    name: str
    B_star: int | None
    selected_poisoned_shards: list[int]
    attacked_cells: list[tuple[int, int]]
    status: int
    status_name: str
    objective: float | None
    y_row: list[int] | None = None
    y_col: list[int] | None = None
    is_optimal: bool = False
    mip_gap: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "B_star": self.B_star,
            "a": self.selected_poisoned_shards,
            "z": self.attacked_cells,
            "status": self.status,
            "status_name": self.status_name,
            "objective": self.objective,
            "y_row": self.y_row,
            "y_col": self.y_col,
            "is_optimal": self.is_optimal,
            "mip_gap": self.mip_gap,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }


def solve_row_stability(
    votes: np.ndarray,
    clean_counts: np.ndarray,
    clean_pred: np.ndarray,
    runner_up: np.ndarray,
    influence: np.ndarray | None = None,
    competitor_mode: str = "all",
) -> CertificateResult:
    """Compatibility stability objective requiring at least one attacked prompt row.

    ``votes`` has shape ``(K, N, L)`` and ``clean_counts`` has shape
    ``(N, L, T)``. Report-facing stability experiments should generally use
    :func:`solve_structured_stability`.
    """
    K, N, L, _ = _validate_stability_shapes(votes, clean_counts, clean_pred, runner_up, influence)
    model, a = _make_model(K, "row_stability")
    z = _add_stability_cell_constraints(model, a, votes, clean_counts, clean_pred, runner_up, influence, competitor_mode=competitor_mode)
    y = model.addVars(N, vtype=GRB.BINARY, name="y_row")
    for i in range(N):
        model.addConstr(y[i] <= gp.quicksum(z[i, j] for j in range(L)), name=f"row_upper_{i}")
        for j in range(L):
            model.addConstr(y[i] >= z[i, j], name=f"row_lower_{i}_{j}")
    model.addConstr(gp.quicksum(y[i] for i in range(N)) >= 1, name="attack_at_least_one_row")
    return _optimize_and_extract("row_stability", model, a, z, y_row=y)


def solve_col_stability(
    votes: np.ndarray,
    clean_counts: np.ndarray,
    clean_pred: np.ndarray,
    runner_up: np.ndarray,
    influence: np.ndarray | None = None,
    competitor_mode: str = "all",
) -> CertificateResult:
    """Compatibility stability objective requiring at least one attacked token column."""
    K, N, L, _ = _validate_stability_shapes(votes, clean_counts, clean_pred, runner_up, influence)
    model, a = _make_model(K, "column_stability")
    z = _add_stability_cell_constraints(model, a, votes, clean_counts, clean_pred, runner_up, influence, competitor_mode=competitor_mode)
    y = model.addVars(L, vtype=GRB.BINARY, name="y_col")
    for j in range(L):
        model.addConstr(y[j] <= gp.quicksum(z[i, j] for i in range(N)), name=f"col_upper_{j}")
        for i in range(N):
            model.addConstr(y[j] >= z[i, j], name=f"col_lower_{i}_{j}")
    model.addConstr(gp.quicksum(y[j] for j in range(L)) >= 1, name="attack_at_least_one_col")
    return _optimize_and_extract("column_stability", model, a, z, y_col=y)


def solve_row_col_stability(
    votes: np.ndarray,
    clean_counts: np.ndarray,
    clean_pred: np.ndarray,
    runner_up: np.ndarray,
    influence: np.ndarray | None = None,
    definition: str = "any_cell",
    competitor_mode: str = "all",
) -> CertificateResult:
    """Compatibility row/column stability wrapper for older named objectives.

    ``definition`` selects legacy objectives such as any cell, full row, full
    column, or full matrix. New report-facing code should use
    :func:`solve_structured_stability` with explicit ``q_rows`` and ``r_cols``.
    """
    K, N, L, _ = _validate_stability_shapes(votes, clean_counts, clean_pred, runner_up, influence)
    model, a = _make_model(K, f"row_col_stability_{definition}")
    z = _add_stability_cell_constraints(model, a, votes, clean_counts, clean_pred, runner_up, influence, competitor_mode=competitor_mode)
    y_row = None
    y_col = None

    if definition == "any_cell":
        model.addConstr(gp.quicksum(z[i, j] for i in range(N) for j in range(L)) >= 1, name="any_cell")
    elif definition == "full_row":
        y_row = model.addVars(N, vtype=GRB.BINARY, name="y_full_row")
        _add_full_row_constraints(model, z, y_row, N, L, q_rows=1)
    elif definition == "full_col":
        y_col = model.addVars(L, vtype=GRB.BINARY, name="y_full_col")
        _add_full_col_constraints(model, z, y_col, N, L)
    elif definition == "full_matrix":
        model.addConstr(gp.quicksum(z[i, j] for i in range(N) for j in range(L)) >= N * L, name="full_matrix")
    else:
        raise ValueError(f"Unknown row/column stability definition: {definition}")

    return _optimize_and_extract(f"row_col_stability_{definition}", model, a, z, y_row=y_row, y_col=y_col)


def solve_structured_stability(
    votes: np.ndarray,
    clean_counts: np.ndarray,
    clean_pred: np.ndarray,
    runner_up: np.ndarray,
    influence: np.ndarray | None = None,
    q_rows: int = 1,
    r_cols: int = 1,
    competitor_mode: str = "all",
) -> CertificateResult:
    """Preferred report-facing structured stability solver.

    The same poisoned-shard variables a[k] are reused across every cell. Use
    q_rows=1,r_cols=1 for one prompt/one token; q_rows=1,r_cols=L for one
    prompt/full sequence; q_rows=N,r_cols=1 for all prompts/one token each; and
    q_rows=N,r_cols=L for the full prompt-token matrix.

    ``competitor_mode="all"`` is exact and checks every non-clean token.
    ``competitor_mode="runner_up"`` is a cheaper DPA-style approximation that
    checks only the original runner-up and may overestimate robustness.
    """
    K, N, L, _ = _validate_stability_shapes(votes, clean_counts, clean_pred, runner_up, influence)
    if q_rows < 1 or q_rows > N:
        raise ValueError(f"q_rows must be in [1, {N}]")
    if r_cols < 1 or r_cols > L:
        raise ValueError(f"r_cols must be in [1, {L}]")

    model, a = _make_model(K, f"row_col_stability_q{q_rows}_r{r_cols}")
    z = _add_stability_cell_constraints(model, a, votes, clean_counts, clean_pred, runner_up, influence, competitor_mode=competitor_mode)
    y_row = model.addVars(N, vtype=GRB.BINARY, name="y_row")
    _add_at_least_r_row_constraints(model, z, y_row, N, L, q_rows=q_rows, r_cols=r_cols)
    return _optimize_and_extract(f"row_col_stability_q{q_rows}_r{r_cols}", model, a, z, y_row=y_row)


def solve_row_validity(
    votes: np.ndarray,
    counts: np.ndarray | None = None,
    target: np.ndarray | None = None,
    T: int | None = None,
    influence: np.ndarray | None = None,
    *,
    clean_counts: np.ndarray | None = None,
) -> CertificateResult:
    """Compatibility validity objective requiring one full harmful prompt row.

    ``votes`` has shape ``(K, N, L)`` and ``counts`` has shape ``(N, L, T)``.
    ``clean_counts`` is accepted as a legacy keyword alias, but validity counts
    need not come from the clean prefix.
    """
    counts, target, T = _resolve_validity_args(counts, clean_counts, target, T)
    K, N, L, T = _validate_validity_shapes(votes, counts, target, T, influence)
    model, a = _make_model(K, "row_validity")
    z = _add_validity_cell_constraints(model, a, votes, counts, target, T, influence)
    y = model.addVars(N, vtype=GRB.BINARY, name="y_row")
    _add_full_row_constraints(model, z, y, N, L, q_rows=1)
    return _optimize_and_extract("row_validity", model, a, z, y_row=y)


def solve_col_validity(
    votes: np.ndarray,
    counts: np.ndarray | None = None,
    target: np.ndarray | None = None,
    T: int | None = None,
    influence: np.ndarray | None = None,
    definition: str = "full_column",
    *,
    clean_counts: np.ndarray | None = None,
) -> CertificateResult:
    """Compatibility validity objective over token columns or any valid cell."""
    counts, target, T = _resolve_validity_args(counts, clean_counts, target, T)
    K, N, L, T = _validate_validity_shapes(votes, counts, target, T, influence)
    model, a = _make_model(K, f"column_validity_{definition}")
    z = _add_validity_cell_constraints(model, a, votes, counts, target, T, influence)
    y_col = None

    if definition == "full_column":
        y_col = model.addVars(L, vtype=GRB.BINARY, name="y_col")
        _add_full_col_constraints(model, z, y_col, N, L)
    elif definition == "any_cell":
        model.addConstr(gp.quicksum(z[i, j] for i in range(N) for j in range(L)) >= 1, name="any_valid_cell")
    else:
        raise ValueError(f"Unknown column validity definition: {definition}")

    return _optimize_and_extract(f"column_validity_{definition}", model, a, z, y_col=y_col)


def solve_row_col_validity(
    votes: np.ndarray,
    counts: np.ndarray | None = None,
    target: np.ndarray | None = None,
    T: int | None = None,
    influence: np.ndarray | None = None,
    q_rows: int = 1,
    definition: str = "at_least_q_rows",
    *,
    clean_counts: np.ndarray | None = None,
) -> CertificateResult:
    """Preferred report-facing shared-allocation validity solver.

    q_rows=1 asks for one full harmful target sequence. q_rows=N asks for full
    harmful target sequences for all prompts. A validity cell succeeds only when
    the harmful target token ties or beats every competitor.
    """
    counts, target, T = _resolve_validity_args(counts, clean_counts, target, T)
    K, N, L, T = _validate_validity_shapes(votes, counts, target, T, influence)
    if q_rows < 1 or q_rows > N:
        raise ValueError(f"q_rows must be in [1, {N}]")
    model, a = _make_model(K, f"row_col_validity_{definition}")
    z = _add_validity_cell_constraints(model, a, votes, counts, target, T, influence)
    y_row = None

    if definition == "at_least_q_rows":
        y_row = model.addVars(N, vtype=GRB.BINARY, name="y_row")
        _add_full_row_constraints(model, z, y_row, N, L, q_rows=q_rows)
        result_name = f"row_col_validity_q{q_rows}"
    elif definition == "full_matrix":
        model.addConstr(gp.quicksum(z[i, j] for i in range(N) for j in range(L)) >= N * L, name="full_matrix")
        result_name = "row_col_validity_full_matrix"
    else:
        raise ValueError(f"Unknown row/column validity definition: {definition}")

    return _optimize_and_extract(result_name, model, a, z, y_row=y_row)


def _make_model(K: int, name: str) -> tuple[gp.Model, gp.tupledict]:
    """Create a silent Gurobi model with binary poisoned-shard variables."""
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model(name, env=env)
    model._env = env
    a = model.addVars(K, vtype=GRB.BINARY, name="a")
    model.setObjective(gp.quicksum(a[k] for k in range(K)), GRB.MINIMIZE)
    return model, a


def _add_stability_cell_constraints(
    model: gp.Model,
    a: gp.tupledict,
    votes: np.ndarray,
    clean_counts: np.ndarray,
    clean_pred: np.ndarray,
    runner_up: np.ndarray,
    influence: np.ndarray | None = None,
    competitor_mode: str = "all",
) -> gp.tupledict:
    """Add per-cell stability attack constraints.

    A stability cell is destabilised if any token other than the clean winner can
    tie or beat the clean winner after poisoning. This checks all competitors,
    not only the original runner-up. The shared a[k] variables are reused across
    all cells, which is the core row/column coupling.

    ``competitor_mode="runner_up"`` restores the cheaper top-vs-second
    simplification for diagnostics.
    """
    K, N, L = votes.shape
    T = clean_counts.shape[-1]
    _validate_competitor_mode(competitor_mode)
    influence = _dense_influence_if_none(influence, K, N, L)
    M = 2 * K + 10
    z = model.addVars(N, L, vtype=GRB.BINARY, name="z_stab")
    u = {}

    for i in range(N):
        for j in range(L):
            w = int(clean_pred[i, j])
            if competitor_mode == "runner_up":
                # DPA-style top-vs-runner-up simplification. This is cheaper but
                # may overestimate robustness if a non-runner-up competitor is
                # easier to promote after poisoning.
                c = int(runner_up[i, j])
                lhs = int(clean_counts[i, j, c]) + gp.quicksum(
                    a[k] * int(influence[k, i, j]) * int(votes[k, i, j] != c) for k in range(K)
                )
                rhs = int(clean_counts[i, j, w]) - gp.quicksum(
                    a[k] * int(influence[k, i, j]) * int(votes[k, i, j] == w) for k in range(K)
                )
                model.addConstr(lhs >= rhs - M * (1 - z[i, j]), name=f"stability_cell_runner_up_{i}_{j}")
                continue
            competitor_vars = []
            for c in range(T):
                if c == w:
                    continue
                u[i, j, c] = model.addVar(vtype=GRB.BINARY, name=f"u_stab_{i}_{j}_{c}")
                competitor_vars.append(u[i, j, c])
                lhs = int(clean_counts[i, j, c]) + gp.quicksum(
                    a[k] * int(influence[k, i, j]) * int(votes[k, i, j] != c) for k in range(K)
                )
                rhs = int(clean_counts[i, j, w]) - gp.quicksum(
                    a[k] * int(influence[k, i, j]) * int(votes[k, i, j] == w) for k in range(K)
                )
                model.addConstr(lhs >= rhs - M * (1 - u[i, j, c]), name=f"stability_cell_{i}_{j}_{c}")
                model.addConstr(u[i, j, c] <= z[i, j], name=f"stability_link_upper_{i}_{j}_{c}")
            model.addConstr(z[i, j] <= gp.quicksum(competitor_vars), name=f"stability_link_lower_{i}_{j}")
    return z


def _add_validity_cell_constraints(
    model: gp.Model,
    a: gp.tupledict,
    votes: np.ndarray,
    counts: np.ndarray,
    target: np.ndarray,
    T: int,
    influence: np.ndarray | None = None,
) -> gp.tupledict:
    """Add per-cell validity attack constraints.

    A validity cell is attacked when the harmful target token ties or beats every
    competitor under the shared poisoned-shard allocation. These counts are the
    validity/harmful-prefix counts, not necessarily clean-prefix counts.
    """
    K, N, L = votes.shape
    influence = _dense_influence_if_none(influence, K, N, L)
    M = 2 * K + 10
    z = model.addVars(N, L, vtype=GRB.BINARY, name="z_val")

    for i in range(N):
        for j in range(L):
            h = int(target[i, j])
            target_count = int(counts[i, j, h]) + gp.quicksum(
                a[k] * int(influence[k, i, j]) * int(votes[k, i, j] != h) for k in range(K)
            )
            for c in range(T):
                if c == h:
                    continue
                competitor_count = int(counts[i, j, c]) - gp.quicksum(
                    a[k] * int(influence[k, i, j]) * int(votes[k, i, j] == c) for k in range(K)
                )
                model.addConstr(target_count >= competitor_count - M * (1 - z[i, j]), name=f"validity_cell_{i}_{j}_{c}")
    return z


def _dense_influence_if_none(influence: np.ndarray | None, K: int, N: int, L: int) -> np.ndarray:
    if influence is None:
        return np.ones((K, N, L), dtype=np.int64)
    if influence.shape != (K, N, L):
        raise ValueError(f"influence must have shape {(K, N, L)}, got {influence.shape}")
    return influence


def _validate_competitor_mode(competitor_mode: str) -> None:
    if competitor_mode not in {"all", "runner_up"}:
        raise ValueError("competitor_mode must be 'all' or 'runner_up'")


def _validate_stability_shapes(
    votes: np.ndarray,
    clean_counts: np.ndarray,
    clean_pred: np.ndarray,
    runner_up: np.ndarray,
    influence: np.ndarray | None,
) -> tuple[int, int, int, int]:
    if votes.ndim != 3:
        raise ValueError(f"votes must have shape (K, N, L), got {votes.shape}")
    K, N, L = votes.shape
    if clean_counts.ndim != 3:
        raise ValueError(f"clean_counts must have shape (N, L, T), got {clean_counts.shape}")
    if clean_counts.shape[:2] != (N, L):
        raise ValueError(f"clean_counts must have leading shape {(N, L)}, got {clean_counts.shape[:2]}")
    T = clean_counts.shape[-1]
    if clean_pred.shape != (N, L):
        raise ValueError(f"clean_pred must have shape {(N, L)}, got {clean_pred.shape}")
    if runner_up.shape != (N, L):
        raise ValueError(f"runner_up must have shape {(N, L)}, got {runner_up.shape}")
    if influence is not None and influence.shape != (K, N, L):
        raise ValueError(f"influence must have shape {(K, N, L)}, got {influence.shape}")
    return K, N, L, T


def _validate_validity_shapes(
    votes: np.ndarray,
    counts: np.ndarray,
    target: np.ndarray,
    T: int,
    influence: np.ndarray | None,
) -> tuple[int, int, int, int]:
    if votes.ndim != 3:
        raise ValueError(f"votes must have shape (K, N, L), got {votes.shape}")
    K, N, L = votes.shape
    if counts.ndim != 3:
        raise ValueError(f"counts must have shape (N, L, T), got {counts.shape}")
    if counts.shape[:2] != (N, L):
        raise ValueError(f"counts must have leading shape {(N, L)}, got {counts.shape[:2]}")
    if T != counts.shape[-1]:
        raise ValueError(f"T must equal counts.shape[-1] ({counts.shape[-1]}), got {T}")
    if target.shape != (N, L):
        raise ValueError(f"target must have shape {(N, L)}, got {target.shape}")
    if influence is not None and influence.shape != (K, N, L):
        raise ValueError(f"influence must have shape {(K, N, L)}, got {influence.shape}")
    return K, N, L, T


def _resolve_validity_args(
    counts: np.ndarray | None,
    clean_counts: np.ndarray | None,
    target: np.ndarray | None,
    T: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    if counts is None:
        counts = clean_counts
    elif clean_counts is not None and clean_counts is not counts:
        raise ValueError("Pass only one of counts or clean_counts.")
    if counts is None:
        raise ValueError("counts is required for validity solvers.")
    if target is None:
        raise ValueError("target is required for validity solvers.")
    if T is None:
        raise ValueError("T is required for validity solvers.")
    return counts, target, T


def _add_full_row_constraints(model: gp.Model, z: gp.tupledict, y_row: gp.tupledict, N: int, L: int, q_rows: int) -> None:
    """Require at least ``q_rows`` prompt rows with all ``L`` token positions attacked."""
    for i in range(N):
        model.addConstr(gp.quicksum(z[i, j] for j in range(L)) >= L * y_row[i], name=f"full_row_sum_{i}")
        for j in range(L):
            model.addConstr(z[i, j] >= y_row[i], name=f"full_row_cell_{i}_{j}")
    model.addConstr(gp.quicksum(y_row[i] for i in range(N)) >= q_rows, name=f"at_least_{q_rows}_rows")


def _add_at_least_r_row_constraints(
    model: gp.Model, z: gp.tupledict, y_row: gp.tupledict, N: int, L: int, q_rows: int, r_cols: int
) -> None:
    """Require at least ``q_rows`` prompt rows, each with at least ``r_cols`` attacked cells."""
    for i in range(N):
        changed_in_row = gp.quicksum(z[i, j] for j in range(L))
        model.addConstr(changed_in_row >= r_cols * y_row[i], name=f"row_{i}_at_least_{r_cols}_lower")
        model.addConstr(changed_in_row <= (r_cols - 1) + (L - r_cols + 1) * y_row[i], name=f"row_{i}_at_least_{r_cols}_upper")
    model.addConstr(gp.quicksum(y_row[i] for i in range(N)) >= q_rows, name=f"at_least_{q_rows}_rows")


def _add_full_col_constraints(model: gp.Model, z: gp.tupledict, y_col: gp.tupledict, N: int, L: int) -> None:
    """Require at least one token position to be attacked for every prompt row."""
    for j in range(L):
        model.addConstr(gp.quicksum(z[i, j] for i in range(N)) >= N * y_col[j], name=f"full_col_sum_{j}")
        for i in range(N):
            model.addConstr(z[i, j] >= y_col[j], name=f"full_col_cell_{i}_{j}")
    model.addConstr(gp.quicksum(y_col[j] for j in range(L)) >= 1, name="at_least_one_col")


def _optimize_and_extract(
    name: str,
    model: gp.Model,
    a: gp.tupledict,
    z: gp.tupledict,
    y_row: gp.tupledict | None = None,
    y_col: gp.tupledict | None = None,
) -> CertificateResult:
    """Optimize a MILP and convert Gurobi status/solution attributes to ``CertificateResult``."""
    # attacked_cells is diagnostic. z is not part of the primary objective, so a
    # feasible optimum may include extra attackable cells beyond those required
    # by the row/column objective. B* remains the certified primary quantity.
    # If Gurobi stops at TIME_LIMIT or SUBOPTIMAL with a feasible solution, B*
    # is the best feasible upper bound, not a certified optimum.
    model.optimize()
    status_name = _status_name(model.status)
    if model.status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT} or model.SolCount == 0:
        return CertificateResult(
            name=name,
            B_star=None,
            selected_poisoned_shards=[],
            attacked_cells=[],
            status=model.status,
            status_name=status_name,
            objective=None,
            is_optimal=False,
            mip_gap=_safe_model_attr(model, "MIPGap"),
            lower_bound=_safe_model_attr(model, "ObjBound"),
            upper_bound=None,
        )

    selected = [k for k in a.keys() if a[k].X > 0.5]
    attacked = [(i, j) for i, j in z.keys() if z[i, j].X > 0.5]
    y_row_values = None if y_row is None else [i for i in y_row.keys() if y_row[i].X > 0.5]
    y_col_values = None if y_col is None else [j for j in y_col.keys() if y_col[j].X > 0.5]
    objective = float(sum(a[k].X for k in a.keys()))
    upper_bound = _safe_model_attr(model, "ObjVal")

    return CertificateResult(
        name=name,
        B_star=int(round(objective)),
        selected_poisoned_shards=selected,
        attacked_cells=attacked,
        status=model.status,
        status_name=status_name,
        objective=objective,
        y_row=y_row_values,
        y_col=y_col_values,
        is_optimal=model.status == GRB.OPTIMAL,
        mip_gap=_safe_model_attr(model, "MIPGap"),
        lower_bound=_safe_model_attr(model, "ObjBound"),
        upper_bound=upper_bound,
    )


def _safe_model_attr(model: gp.Model, attr_name: str) -> float | None:
    try:
        value = getattr(model, attr_name)
    except (gp.GurobiError, AttributeError):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _status_name(status: int) -> str:
    status_map = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return status_map.get(status, f"STATUS_{status}")
