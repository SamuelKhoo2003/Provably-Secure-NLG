from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ortools.sat.python import cp_model


@dataclass
class ILPResult:
    selected_indices: List[int]
    objective_value: float


def select_poison_indices(scores: List[float], budget: int) -> ILPResult:
    """Select up to `budget` indices maximizing sum(scores[i] * z_i)."""
    if budget <= 0 or not scores:
        return ILPResult(selected_indices=[], objective_value=0.0)

    model = cp_model.CpModel()
    n = len(scores)
    z = [model.NewBoolVar(f"z_{i}") for i in range(n)]

    model.Add(sum(z) <= min(budget, n))

    scale = 1000
    int_scores = [int(round(s * scale)) for s in scores]
    model.Maximize(sum(int_scores[i] * z[i] for i in range(n)))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return ILPResult(selected_indices=[], objective_value=0.0)

    selected = [i for i in range(n) if solver.Value(z[i]) == 1]
    obj = solver.ObjectiveValue() / scale
    return ILPResult(selected_indices=selected, objective_value=obj)
