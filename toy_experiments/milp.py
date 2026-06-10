"""Gurobi MILP solvers for shared poisoning certificates."""

from __future__ import annotations

from dataclasses import dataclass
import os

import gurobipy as gp
import numpy as np
from gurobipy import GRB


@dataclass(frozen=True)
class CertificateResult:
    """Result returned by a certificate MILP solve."""

    name: str
    B_star: int | None
    selected_poisoned_shards: list[int]
    attacked_cells: list[tuple[int, int]]
    status: int
    status_name: str
    objective: float | None
    y_row: list[int] | None = None
    is_optimal: bool = False
    mip_gap: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None


def solve_structured_stability(
    votes: np.ndarray,
    clean_counts: np.ndarray,
    clean_pred: np.ndarray,
    influence: np.ndarray | None = None,
    q_rows: int = 1,
    r_cols: int = 1,
    gurobi_threads: int | None = None,
) -> CertificateResult:
    """Minimize poisoned shards needed to destabilize q rows by r token positions."""
    K, N, L, _ = _validate_stability_shapes(votes, clean_counts, clean_pred, influence)
    if q_rows < 1 or q_rows > N:
        raise ValueError(f"q_rows must be in [1, {N}]")
    if r_cols < 1 or r_cols > L:
        raise ValueError(f"r_cols must be in [1, {L}]")

    model, a = _make_model(K, f"row_col_stability_q{q_rows}_r{r_cols}", gurobi_threads=gurobi_threads)
    z = _add_stability_cell_constraints(model, a, votes, clean_counts, clean_pred, influence)
    y_row = model.addVars(N, vtype=GRB.BINARY, name="y_row")
    _add_at_least_r_row_constraints(model, z, y_row, N, L, q_rows=q_rows, r_cols=r_cols)
    return _optimize_and_extract(f"row_col_stability_q{q_rows}_r{r_cols}", model, a, z, y_row=y_row)


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
    gurobi_threads: int | None = None,
) -> CertificateResult:
    """Minimize poisoned shards needed to force harmful target sequences."""
    counts, target, T = _resolve_validity_args(counts, clean_counts, target, T)
    K, N, L, _ = _validate_validity_shapes(votes, counts, target, T, influence)
    if q_rows < 1 or q_rows > N:
        raise ValueError(f"q_rows must be in [1, {N}]")

    model, a = _make_model(K, f"row_col_validity_{definition}", gurobi_threads=gurobi_threads)
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


def resolve_gurobi_threads(gurobi_threads: int | None = None) -> int:
    """Resolve Gurobi Threads with explicit argument > env var > default 0.

    ``0`` is Gurobi automatic mode. Positive integers request a fixed maximum
    thread count. Negative values are intentionally rejected.
    """
    if gurobi_threads is not None:
        return _validate_gurobi_threads(gurobi_threads, "gurobi_threads")
    env_value = os.environ.get("GUROBI_THREADS")
    if env_value is not None:
        return _validate_gurobi_threads(env_value, "GUROBI_THREADS")
    return 0


def _validate_gurobi_threads(value: object, source: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source} must be an integer >= 0")
    if isinstance(value, int):
        threads = value
    elif isinstance(value, str):
        try:
            threads = int(value)
        except ValueError as exc:
            raise ValueError(f"{source} must be an integer >= 0") from exc
    else:
        raise ValueError(f"{source} must be an integer >= 0")
    if threads < 0:
        raise ValueError(f"{source} must be >= 0; use 0 for Gurobi automatic mode")
    return threads


def _make_model(K: int, name: str, gurobi_threads: int | None = None) -> tuple[gp.Model, gp.tupledict]:
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model(name, env=env)
    model.setParam("Threads", resolve_gurobi_threads(gurobi_threads))
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
    influence: np.ndarray | None,
) -> gp.tupledict:
    K, N, L = votes.shape
    T = clean_counts.shape[-1]
    influence = _dense_influence_if_none(influence, K, N, L)
    big_m = 2 * K + 10
    z = model.addVars(N, L, vtype=GRB.BINARY, name="z_stab")
    u = {}

    for i in range(N):
        for j in range(L):
            winner = int(clean_pred[i, j])
            competitor_vars = []
            for competitor in range(T):
                if competitor == winner:
                    continue
                u[i, j, competitor] = model.addVar(vtype=GRB.BINARY, name=f"u_stab_{i}_{j}_{competitor}")
                competitor_vars.append(u[i, j, competitor])
                lhs = int(clean_counts[i, j, competitor]) + gp.quicksum(
                    a[k] * int(influence[k, i, j]) * int(votes[k, i, j] != competitor) for k in range(K)
                )
                rhs = int(clean_counts[i, j, winner]) - gp.quicksum(
                    a[k] * int(influence[k, i, j]) * int(votes[k, i, j] == winner) for k in range(K)
                )
                model.addConstr(lhs >= rhs - big_m * (1 - u[i, j, competitor]), name=f"stability_cell_{i}_{j}_{competitor}")
                model.addConstr(u[i, j, competitor] <= z[i, j], name=f"stability_link_upper_{i}_{j}_{competitor}")
            model.addConstr(z[i, j] <= gp.quicksum(competitor_vars), name=f"stability_link_lower_{i}_{j}")
    return z


def _add_validity_cell_constraints(
    model: gp.Model,
    a: gp.tupledict,
    votes: np.ndarray,
    counts: np.ndarray,
    target: np.ndarray,
    T: int,
    influence: np.ndarray | None,
) -> gp.tupledict:
    K, N, L = votes.shape
    influence = _dense_influence_if_none(influence, K, N, L)
    big_m = 2 * K + 10
    z = model.addVars(N, L, vtype=GRB.BINARY, name="z_val")

    for i in range(N):
        for j in range(L):
            harmful = int(target[i, j])
            target_count = int(counts[i, j, harmful]) + gp.quicksum(
                a[k] * int(influence[k, i, j]) * int(votes[k, i, j] != harmful) for k in range(K)
            )
            for competitor in range(T):
                if competitor == harmful:
                    continue
                competitor_count = int(counts[i, j, competitor]) - gp.quicksum(
                    a[k] * int(influence[k, i, j]) * int(votes[k, i, j] == competitor) for k in range(K)
                )
                model.addConstr(target_count >= competitor_count - big_m * (1 - z[i, j]), name=f"validity_cell_{i}_{j}_{competitor}")
    return z


def _add_full_row_constraints(model: gp.Model, z: gp.tupledict, y_row: gp.tupledict, N: int, L: int, q_rows: int) -> None:
    for i in range(N):
        model.addConstr(gp.quicksum(z[i, j] for j in range(L)) >= L * y_row[i], name=f"full_row_sum_{i}")
        for j in range(L):
            model.addConstr(z[i, j] >= y_row[i], name=f"full_row_cell_{i}_{j}")
    model.addConstr(gp.quicksum(y_row[i] for i in range(N)) >= q_rows, name=f"at_least_{q_rows}_rows")


def _add_at_least_r_row_constraints(
    model: gp.Model,
    z: gp.tupledict,
    y_row: gp.tupledict,
    N: int,
    L: int,
    q_rows: int,
    r_cols: int,
) -> None:
    for i in range(N):
        changed_in_row = gp.quicksum(z[i, j] for j in range(L))
        model.addConstr(changed_in_row >= r_cols * y_row[i], name=f"row_{i}_at_least_{r_cols}_lower")
        model.addConstr(changed_in_row <= (r_cols - 1) + (L - r_cols + 1) * y_row[i], name=f"row_{i}_at_least_{r_cols}_upper")
    model.addConstr(gp.quicksum(y_row[i] for i in range(N)) >= q_rows, name=f"at_least_{q_rows}_rows")


def _optimize_and_extract(
    name: str,
    model: gp.Model,
    a: gp.tupledict,
    z: gp.tupledict,
    y_row: gp.tupledict | None = None,
) -> CertificateResult:
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
    objective = float(sum(a[k].X for k in a.keys()))
    return CertificateResult(
        name=name,
        B_star=int(round(objective)),
        selected_poisoned_shards=selected,
        attacked_cells=attacked,
        status=model.status,
        status_name=status_name,
        objective=objective,
        y_row=y_row_values,
        is_optimal=model.status == GRB.OPTIMAL,
        mip_gap=_safe_model_attr(model, "MIPGap"),
        lower_bound=_safe_model_attr(model, "ObjBound"),
        upper_bound=_safe_model_attr(model, "ObjVal"),
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


def _dense_influence_if_none(influence: np.ndarray | None, K: int, N: int, L: int) -> np.ndarray:
    if influence is None:
        return np.ones((K, N, L), dtype=np.int64)
    if influence.shape != (K, N, L):
        raise ValueError(f"influence must have shape {(K, N, L)}, got {influence.shape}")
    return influence


def _validate_stability_shapes(
    votes: np.ndarray,
    clean_counts: np.ndarray,
    clean_pred: np.ndarray,
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
    _validate_counts(clean_counts, "clean_counts")
    _validate_token_ids(votes, T, "votes")
    if clean_pred.shape != (N, L):
        raise ValueError(f"clean_pred must have shape {(N, L)}, got {clean_pred.shape}")
    _validate_token_ids(clean_pred, T, "clean_pred")
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
    _validate_counts(counts, "counts")
    _validate_token_ids(votes, T, "votes")
    if target.shape != (N, L):
        raise ValueError(f"target must have shape {(N, L)}, got {target.shape}")
    _validate_token_ids(target, T, "target")
    if influence is not None and influence.shape != (K, N, L):
        raise ValueError(f"influence must have shape {(K, N, L)}, got {influence.shape}")
    return K, N, L, T


def _validate_counts(counts: np.ndarray, name: str) -> None:
    if np.any(counts < 0):
        raise ValueError(f"{name} must be non-negative")


def _validate_token_ids(tokens: np.ndarray, T: int, name: str) -> None:
    if np.any(tokens < 0) or np.any(tokens >= T):
        raise ValueError(f"{name} must contain token ids in [0, {T})")


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
