from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gurobipy as gp
import numpy as np
from gurobipy import GRB


@dataclass(frozen=True)
class CertificateResult:
    name: str
    B_star: int | None
    selected_poisoned_shards: list[int]
    attacked_cells: list[tuple[int, int]]
    status: int
    status_name: str
    objective: float | None
    y_row: list[int] | None = None
    y_col: list[int] | None = None

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
        }


def solve_row_stability(
    votes: np.ndarray, clean_counts: np.ndarray, clean_pred: np.ndarray, runner_up: np.ndarray, influence: np.ndarray | None = None
) -> CertificateResult:
    K, N, L = votes.shape
    model, a = _make_model(K, "row_stability")
    z = _add_stability_cell_constraints(model, a, votes, clean_counts, clean_pred, runner_up, influence)
    y = model.addVars(N, vtype=GRB.BINARY, name="y_row")
    for i in range(N):
        model.addConstr(y[i] <= gp.quicksum(z[i, j] for j in range(L)), name=f"row_upper_{i}")
        for j in range(L):
            model.addConstr(y[i] >= z[i, j], name=f"row_lower_{i}_{j}")
    model.addConstr(gp.quicksum(y[i] for i in range(N)) >= 1, name="attack_at_least_one_row")
    return _optimize_and_extract("row_stability", model, a, z, y_row=y)


def solve_col_stability(
    votes: np.ndarray, clean_counts: np.ndarray, clean_pred: np.ndarray, runner_up: np.ndarray, influence: np.ndarray | None = None
) -> CertificateResult:
    K, N, L = votes.shape
    model, a = _make_model(K, "column_stability")
    z = _add_stability_cell_constraints(model, a, votes, clean_counts, clean_pred, runner_up, influence)
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
) -> CertificateResult:
    K, N, L = votes.shape
    model, a = _make_model(K, f"row_col_stability_{definition}")
    z = _add_stability_cell_constraints(model, a, votes, clean_counts, clean_pred, runner_up, influence)
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
) -> CertificateResult:
    """Require at least q rows, each with at least r destabilised token cells."""
    K, N, L = votes.shape
    if q_rows < 1 or q_rows > N:
        raise ValueError(f"q_rows must be in [1, {N}]")
    if r_cols < 1 or r_cols > L:
        raise ValueError(f"r_cols must be in [1, {L}]")

    model, a = _make_model(K, f"row_col_stability_q{q_rows}_r{r_cols}")
    z = _add_stability_cell_constraints(model, a, votes, clean_counts, clean_pred, runner_up, influence)
    y_row = model.addVars(N, vtype=GRB.BINARY, name="y_row")
    _add_at_least_r_row_constraints(model, z, y_row, N, L, q_rows=q_rows, r_cols=r_cols)
    return _optimize_and_extract(f"row_col_stability_q{q_rows}_r{r_cols}", model, a, z, y_row=y_row)


def solve_row_validity(votes: np.ndarray, clean_counts: np.ndarray, target: np.ndarray, T: int, influence: np.ndarray | None = None) -> CertificateResult:
    K, N, L = votes.shape
    model, a = _make_model(K, "row_validity")
    z = _add_validity_cell_constraints(model, a, votes, clean_counts, target, T, influence)
    y = model.addVars(N, vtype=GRB.BINARY, name="y_row")
    _add_full_row_constraints(model, z, y, N, L, q_rows=1)
    return _optimize_and_extract("row_validity", model, a, z, y_row=y)


def solve_col_validity(
    votes: np.ndarray, clean_counts: np.ndarray, target: np.ndarray, T: int, influence: np.ndarray | None = None, definition: str = "full_column"
) -> CertificateResult:
    K, N, L = votes.shape
    model, a = _make_model(K, f"column_validity_{definition}")
    z = _add_validity_cell_constraints(model, a, votes, clean_counts, target, T, influence)
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
    clean_counts: np.ndarray,
    target: np.ndarray,
    T: int,
    influence: np.ndarray | None = None,
    q_rows: int = 1,
    definition: str = "at_least_q_rows",
) -> CertificateResult:
    K, N, L = votes.shape
    if q_rows < 1 or q_rows > N:
        raise ValueError(f"q_rows must be in [1, {N}]")
    model, a = _make_model(K, f"row_col_validity_{definition}")
    z = _add_validity_cell_constraints(model, a, votes, clean_counts, target, T, influence)
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
) -> gp.tupledict:
    K, N, L = votes.shape
    influence = _dense_influence_if_none(influence, K, N, L)
    M = 2 * K + 10
    z = model.addVars(N, L, vtype=GRB.BINARY, name="z_stab")

    for i in range(N):
        for j in range(L):
            w = int(clean_pred[i, j])
            c = int(runner_up[i, j])
            lhs = int(clean_counts[i, j, c]) + gp.quicksum(
                a[k] * int(influence[k, i, j]) * int(votes[k, i, j] != c) for k in range(K)
            )
            rhs = int(clean_counts[i, j, w]) - gp.quicksum(
                a[k] * int(influence[k, i, j]) * int(votes[k, i, j] == w) for k in range(K)
            )
            model.addConstr(lhs >= rhs - M * (1 - z[i, j]), name=f"stability_cell_{i}_{j}")
    return z


def _add_validity_cell_constraints(
    model: gp.Model,
    a: gp.tupledict,
    votes: np.ndarray,
    clean_counts: np.ndarray,
    target: np.ndarray,
    T: int,
    influence: np.ndarray | None = None,
) -> gp.tupledict:
    K, N, L = votes.shape
    influence = _dense_influence_if_none(influence, K, N, L)
    M = 2 * K + 10
    z = model.addVars(N, L, vtype=GRB.BINARY, name="z_val")

    for i in range(N):
        for j in range(L):
            h = int(target[i, j])
            target_count = int(clean_counts[i, j, h]) + gp.quicksum(
                a[k] * int(influence[k, i, j]) * int(votes[k, i, j] != h) for k in range(K)
            )
            for c in range(T):
                if c == h:
                    continue
                competitor_count = int(clean_counts[i, j, c]) - gp.quicksum(
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


def _add_full_row_constraints(model: gp.Model, z: gp.tupledict, y_row: gp.tupledict, N: int, L: int, q_rows: int) -> None:
    for i in range(N):
        model.addConstr(gp.quicksum(z[i, j] for j in range(L)) >= L * y_row[i], name=f"full_row_sum_{i}")
        for j in range(L):
            model.addConstr(z[i, j] >= y_row[i], name=f"full_row_cell_{i}_{j}")
    model.addConstr(gp.quicksum(y_row[i] for i in range(N)) >= q_rows, name=f"at_least_{q_rows}_rows")


def _add_at_least_r_row_constraints(
    model: gp.Model, z: gp.tupledict, y_row: gp.tupledict, N: int, L: int, q_rows: int, r_cols: int
) -> None:
    for i in range(N):
        changed_in_row = gp.quicksum(z[i, j] for j in range(L))
        model.addConstr(changed_in_row >= r_cols * y_row[i], name=f"row_{i}_at_least_{r_cols}_lower")
        model.addConstr(changed_in_row <= (r_cols - 1) + (L - r_cols + 1) * y_row[i], name=f"row_{i}_at_least_{r_cols}_upper")
    model.addConstr(gp.quicksum(y_row[i] for i in range(N)) >= q_rows, name=f"at_least_{q_rows}_rows")


def _add_full_col_constraints(model: gp.Model, z: gp.tupledict, y_col: gp.tupledict, N: int, L: int) -> None:
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
        )

    selected = [k for k in a.keys() if a[k].X > 0.5]
    attacked = [(i, j) for i, j in z.keys() if z[i, j].X > 0.5]
    y_row_values = None if y_row is None else [i for i in y_row.keys() if y_row[i].X > 0.5]
    y_col_values = None if y_col is None else [j for j in y_col.keys() if y_col[j].X > 0.5]
    objective = float(model.ObjVal)

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
    )


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
