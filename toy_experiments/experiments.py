"""Experiment CLI, benchmark generation, baselines, and PDF plotting.

This module is the orchestration layer for the first-party toy certificate
implementation. It builds synthetic :class:`toy_experiments.data.ToyData`
instances, calls the shared MILP solvers, writes benchmark CSVs, computes
DPA/TPA/atomic-phrase/independent-composition baselines, and renders PDF plots.
The external ``phd_reference/`` tree is not imported or modified here.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import os
from pathlib import Path
import tempfile
from time import perf_counter

_PLOT_CACHE_DIR = Path(tempfile.gettempdir()) / "provably-secure-nlg-plot-cache"
_PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .baselines import (
    aggregate_plain_dpa_sequence_baselines,
    aggregate_tpa_sequence_baselines,
    atomic_phrase_validity_row_budgets as _atomic_phrase_validity_row_budgets,
    cell_stability_budgets as _cell_stability_budgets,
    cell_validity_budgets as _cell_validity_budgets,
    compute_reference_baselines,
    phd_margin_stability_budgets as _phd_margin_stability_budgets,
    plain_dpa_validity_token_budgets as _plain_dpa_validity_token_budgets,
    targeted_validity_token_budgets as _targeted_validity_token_budgets,
)
from .csv_io import (
    read_optional_csv as _read_optional_csv_raw,
    read_rows_csv as _read_rows_csv_raw,
    write_rows_csv as _write_rows_csv,
)
from .data import ToyData, generate_toy_votes, generate_validity_demo_votes, slice_toy_data, stability_margins
from .milp import (
    CertificateResult,
    resolve_gurobi_threads,
    solve_row_col_validity,
    solve_structured_stability,
)


class ConfigError(ValueError):
    """Raised when a benchmark YAML config is missing or malformed."""


TOY_METHOD_ALIASES = {
    "TPA max-token phrase blocker": "TPA max-token phrase baseline",
    "TPA multi-sample validity": "TPA max-token phrase baseline",
}


def _canonicalize_tpa_method_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Normalize legacy count-based TPA labels when reading old CSVs."""
    normalized = []
    for row in rows:
        canonical = dict(row)
        method = canonical.get("method")
        if isinstance(method, str):
            canonical["method"] = TOY_METHOD_ALIASES.get(method, method)
        normalized.append(canonical)
    return normalized


def _read_rows_csv(path: Path) -> list[dict[str, object]]:
    return _canonicalize_tpa_method_rows(_read_rows_csv_raw(path))


def _read_optional_csv(path: Path) -> list[dict[str, object]]:
    return _canonicalize_tpa_method_rows(_read_optional_csv_raw(path))


def visualize_instance(
    K: int,
    N: int,
    L: int,
    T: int,
    delta_stab: float,
    delta_val: float,
    target_bias: float,
    seed: int | Iterable[int],
    influence_mode: str,
    save_dir: str,
) -> None:
    """Generate and save one instance-level visualization bundle."""
    data = generate_toy_votes(
        K=K, N=N, L=L, T=T, delta_stab=delta_stab, delta_val=delta_val, target_bias=target_bias, seed=seed, influence_mode=influence_mode
    )
    _print_instance_summary(data, K, N, L, T, delta_stab, delta_val, target_bias, seed, influence_mode)
    print_console_grid(data)
    save_instance_plots(
        data,
        Path(save_dir),
        K=K,
        N=N,
        L=L,
        T=T,
        delta_stab=delta_stab,
        delta_val=delta_val,
        target_bias=target_bias,
        seed=seed,
        influence_mode=influence_mode,
    )


def benchmark_scale(
    Ks: Iterable[int],
    Ns: Iterable[int],
    Ls: Iterable[int],
    Ts: Iterable[int],
    delta_stabs: Iterable[float],
    delta_vals: Iterable[float],
    target_biases: Iterable[float],
    influence_mode: str,
    seed: int,
    save_dir: str,
    make_plots: bool = False,
    budget_max: int = 15,
    make_budget_curves: bool = True,
    objective_family: str = "full",
    make_stability_objectives: bool | None = None,
    make_validity_objectives: bool | None = None,
    make_stability_budget_curves: bool | None = None,
    make_validity_budget_curves: bool | None = None,
    dry_run: bool = False,
    verbose: bool = False,
    generator: str = "toy",
    group_size: int = 3,
    target_gap: int = 1,
    overlap: int = 0,
    validity_demo_distribution: str = "deterministic",
    num_competitor_min: int = 2,
    num_competitor_max: int = 8,
    target_count_min: int = 0,
    target_count_max: int = 3,
    competitor_gap_min: int = 2,
    competitor_gap_max: int = 6,
    competitor_jitter: int = 2,
    row_difficulty_jitter: bool = True,
    position_difficulty_jitter: bool = True,
    budget_plot_num_points: int | None = None,
    gurobi_threads: int | None = None,
) -> list[dict[str, object]]:
    """Generate benchmark CSV rows for the configured parameter grid."""
    Ks, Ns, Ls, Ts = list(Ks), list(Ns), list(Ls), list(Ts)
    seeds = [seed] if isinstance(seed, int) else list(seed)
    if not seeds or not all(isinstance(value, int) and not isinstance(value, bool) for value in seeds):
        raise ValueError("seed must be an integer or a non-empty iterable of integers")
    delta_stabs, delta_vals, target_biases = list(delta_stabs), list(delta_vals), list(target_biases)
    objective_flags = _resolve_objective_flags(
        objective_family=objective_family,
        make_budget_curves=make_budget_curves,
        make_stability_objectives=make_stability_objectives,
        make_validity_objectives=make_validity_objectives,
        make_stability_budget_curves=make_stability_budget_curves,
        make_validity_budget_curves=make_validity_budget_curves,
    )
    if dry_run:
        _print_benchmark_dry_run(
            Ks=Ks,
            Ns=Ns,
            Ls=Ls,
            Ts=Ts,
            delta_stabs=delta_stabs,
            delta_vals=delta_vals,
            target_biases=target_biases,
            generator=generator,
            objective_family=objective_family,
            objective_flags=objective_flags,
            budget_max=budget_max,
            save_dir=save_dir,
            verbose=verbose,
            group_size=group_size,
            overlap=overlap,
            seeds=seeds,
        )
        return []
    resolved_gurobi_threads = resolve_gurobi_threads(gurobi_threads)
    print(f"Gurobi Threads = {resolved_gurobi_threads}")
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    budget_curve_rows: list[dict[str, object]] = []
    K_master_requested = max(Ks)
    N_master = max(Ns)
    L_master = max(Ls)
    T_master = max(Ts)
    K_master = (
        max(K_master_requested, _validity_demo_min_required_shards(L_master, group_size, overlap))
        if generator == "validity_demo"
        else K_master_requested
    )
    master_cache: dict[tuple[int, float, float, float], ToyData] = {}

    for current_seed in seeds:
        for K in Ks:
            for N in Ns:
                for L in Ls:
                    for T in Ts:
                        for delta_stab in delta_stabs:
                            for delta_val in delta_vals:
                                for target_bias in target_biases:
                                    group_key = (current_seed, delta_stab, delta_val, target_bias)
                                    master = master_cache.get(group_key)
                                    if master is None:
                                        if generator == "validity_demo":
                                            master = generate_validity_demo_votes(
                                                L=L_master,
                                                group_size=group_size,
                                                target_gap=target_gap,
                                                overlap=overlap,
                                                N=N_master,
                                                T=T_master,
                                                seed=current_seed,
                                                K=K_master,
                                                distribution=validity_demo_distribution,
                                                num_competitor_min=num_competitor_min,
                                                num_competitor_max=num_competitor_max,
                                                target_count_min=target_count_min,
                                                target_count_max=target_count_max,
                                                competitor_gap_min=competitor_gap_min,
                                                competitor_gap_max=competitor_gap_max,
                                                competitor_jitter=competitor_jitter,
                                                row_difficulty_jitter=row_difficulty_jitter,
                                                position_difficulty_jitter=position_difficulty_jitter,
                                                minimum_requested_K=min(Ks),
                                            )
                                        elif generator == "toy":
                                            master = generate_toy_votes(
                                                K=K_master,
                                                N=N_master,
                                                L=L_master,
                                                T=T_master,
                                                delta_stab=delta_stab,
                                                delta_val=delta_val,
                                                target_bias=target_bias,
                                                seed=current_seed,
                                                influence_mode=influence_mode,
                                            )
                                        else:
                                            raise ValueError(f"Unknown generator: {generator}")
                                        master_cache[group_key] = master

                                    if generator == "validity_demo":
                                        K_actual = max(K, _validity_demo_min_required_shards(L, group_size, overlap))
                                    else:
                                        K_actual = K
                                    data = slice_toy_data(master, K=K_actual, N=N, L=L, T=T)
                                    data.metadata.update({"K_requested": K, "K_actual": K_actual})
                                    start = perf_counter()
                                    results = _solve_benchmark_certificates(
                                        data,
                                        T,
                                        make_stability_objectives=objective_flags["make_stability_objectives"],
                                        make_validity_objectives=objective_flags["make_validity_objectives"],
                                        gurobi_threads=resolved_gurobi_threads,
                                    )
                                    runtime_total = perf_counter() - start
                                    row = {
                                        "K": K_actual,
                                        "K_requested": K,
                                        "K_actual": K_actual,
                                        "K_master": K_master,
                                        "N_master": N_master,
                                        "L_master": L_master,
                                        "T_master": T_master,
                                        "coupled_generation": True,
                                        "N": N,
                                        "L": L,
                                        "T": T,
                                        "generator": generator,
                                        "delta_stab": delta_stab,
                                        "delta_val": delta_val,
                                        "target_bias": target_bias,
                                        "seed": current_seed,
                                        "influence_mode": influence_mode,
                                        "runtime_gurobi_total": f"{runtime_total:.6f}",
                                    }
                                    if generator == "validity_demo":
                                        row.update(
                                            {
                                                "group_size": group_size,
                                                "overlap": overlap,
                                                "validity_demo_distribution": validity_demo_distribution,
                                                "budget_plot_num_points": budget_plot_num_points or "",
                                            }
                                        )
                                        if validity_demo_distribution == "deterministic":
                                            row["target_gap"] = target_gap
                                    for result in results:
                                        metric_name = _csv_metric_name(result.name, N, L)
                                        _add_certificate_columns(row, metric_name, result)
                                    _fill_degenerate_corner_columns(row)
                                    row.update(
                                        _compute_benchmark_baselines(
                                            data,
                                            make_stability_objectives=objective_flags["make_stability_objectives"],
                                            make_validity_objectives=objective_flags["make_validity_objectives"],
                                        )
                                    )
                                    if generator == "validity_demo":
                                        row.update(_validity_demo_diagnostics(data))
                                    _add_validity_demo_gap_columns(row)
                                    rows.append(row)
                                    budgets = list(range(0, min(K_actual, budget_max) + 1))
                                    metadata = _benchmark_metadata(
                                        seed=current_seed,
                                        K=K_actual,
                                        N=N,
                                        L=L,
                                        T=T,
                                        delta_stab=delta_stab,
                                        delta_val=delta_val,
                                        target_bias=target_bias,
                                        influence_mode=influence_mode,
                                    )
                                    metadata.update(
                                        {
                                            "coupled_generation": True,
                                            "K_requested": K,
                                            "K_actual": K_actual,
                                            "K_master": K_master,
                                            "N_master": N_master,
                                            "L_master": L_master,
                                            "T_master": T_master,
                                        }
                                    )
                                    if objective_flags["make_stability_budget_curves"] or objective_flags["make_validity_budget_curves"]:
                                        budget_curve_rows.extend(
                                            compute_radius_derived_budget_curve_rows(
                                                data,
                                                T=T,
                                                budgets=budgets,
                                                metadata=metadata,
                                                make_stability_curves=objective_flags["make_stability_budget_curves"],
                                                make_validity_curves=objective_flags["make_validity_budget_curves"],
                                                gurobi_threads=resolved_gurobi_threads,
                                            )
                                        )
                                    print(
                                        "bench "
                                        f"seed={current_seed} K={K_actual} N={N} L={L} T={T} "
                                        f"delta_stab={delta_stab} delta_val={delta_val} target_bias={target_bias}: "
                                        + ", ".join(f"{result.name}={result.B_star}" for result in results)
                                    )

    csv_path = output_dir / "benchmark_results.csv"
    _write_rows_csv(csv_path, rows)
    print()
    print(f"Wrote benchmark CSV: {csv_path}")
    if len(seeds) > 1:
        aggregate_path = output_dir / "benchmark_results_seed_aggregate.csv"
        _write_rows_csv(aggregate_path, _aggregate_result_rows_across_seeds(rows))
        print(f"Wrote seed-aggregated benchmark CSV: {aggregate_path}")
    if objective_flags["make_stability_budget_curves"] or objective_flags["make_validity_budget_curves"]:
        budget_csv_path = output_dir / "benchmark_budget_curves.csv"
        _write_rows_csv(budget_csv_path, budget_curve_rows)
        print(f"Wrote budget-curve CSV: {budget_csv_path}")
        if len(seeds) > 1:
            aggregate_budget_path = output_dir / "benchmark_budget_curves_seed_aggregate.csv"
            _write_rows_csv(aggregate_budget_path, _aggregate_budget_rows_across_seeds(budget_curve_rows))
            print(f"Wrote seed-aggregated budget-curve CSV: {aggregate_budget_path}")
    if make_plots:
        save_default_report_plots(rows, output_dir, csv_path=csv_path)
        print(f"Wrote benchmark plots under: {output_dir}")
    else:
        print("Skipped plots. Generate plots with: ./toy_experiments/scripts/plot.sh")
    return rows


RESULT_SEED_GROUP_COLUMNS = (
    "K",
    "K_requested",
    "K_actual",
    "K_master",
    "N_master",
    "L_master",
    "T_master",
    "coupled_generation",
    "N",
    "L",
    "T",
    "generator",
    "delta_stab",
    "delta_val",
    "target_bias",
    "influence_mode",
    "group_size",
    "overlap",
    "validity_demo_distribution",
    "target_gap",
    "budget_plot_num_points",
)

BUDGET_SEED_MEASURE_COLUMNS = (
    "certified_fraction",
    "attacked_fraction",
    "mean_radius",
    "median_radius",
    "min_radius",
    "max_radius",
    "num_certified",
    "num_known",
    "num_unknown",
    "num_total",
)


def _aggregate_result_rows_across_seeds(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Average benchmark measurements across seeds and retain observed ranges."""
    return _aggregate_rows_across_seeds(rows, RESULT_SEED_GROUP_COLUMNS, measure_columns=None)


def _aggregate_budget_rows_across_seeds(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Average budget-curve measurements across seeds and retain observed ranges."""
    group_columns = tuple(
        key
        for key in rows[0]
        if key not in {"seed", *BUDGET_SEED_MEASURE_COLUMNS}
    ) if rows else ()
    return _aggregate_rows_across_seeds(rows, group_columns, measure_columns=BUDGET_SEED_MEASURE_COLUMNS)


def _aggregate_rows_across_seeds(
    rows: list[dict[str, object]],
    group_columns: Iterable[str],
    *,
    measure_columns: Iterable[str] | None,
) -> list[dict[str, object]]:
    group_columns = tuple(group_columns)
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(key, "") for key in group_columns), []).append(row)

    output: list[dict[str, object]] = []
    for key, group in grouped.items():
        aggregate = dict(zip(group_columns, key))
        seeds = sorted(
            {
                int(seed)
                for row in group
                if (seed := _numeric_value(row.get("seed"))) is not None
            }
        )
        aggregate["seed_count"] = len(seeds)
        aggregate["seed_values"] = ",".join(str(seed) for seed in seeds)
        candidates = tuple(measure_columns) if measure_columns is not None else tuple(
            column
            for column in group[0]
            if column not in {*group_columns, "seed"} and not column.endswith(("_status", "_is_optimal"))
        )
        for column in candidates:
            values = [
                value
                for row in group
                if (value := _numeric_value(row.get(column))) is not None
            ]
            if not values:
                continue
            mean_value = float(np.mean(values))
            min_value = float(np.min(values))
            max_value = float(np.max(values))
            aggregate[column] = mean_value
            aggregate[f"{column}_min"] = min_value
            aggregate[f"{column}_max"] = max_value
            aggregate[f"{column}_minus"] = mean_value - min_value
            aggregate[f"{column}_plus"] = max_value - mean_value
        for column in group[0]:
            if not column.endswith(("_status", "_is_optimal")):
                continue
            values = {row.get(column, "") for row in group}
            aggregate[column] = values.pop() if len(values) == 1 else "MIXED"
        output.append(aggregate)
    return output


def _csv_metric_name(result_name: str, N: int, L: int) -> str:
    if result_name.startswith("row_col_stability_q"):
        q_part, r_part = result_name.removeprefix("row_col_stability_q").split("_r", maxsplit=1)
        q_rows = int(q_part)
        r_cols = int(r_part)
        q_label = "q1" if q_rows == 1 else ("qN" if q_rows == N else f"q{q_rows}")
        r_label = "r1" if r_cols == 1 else ("rL" if r_cols == L else f"r{r_cols}")
        return f"row_col_stab_{q_label}_{r_label}"
    if result_name == "row_col_validity_q1":
        return "row_col_val_q1"
    if result_name == f"row_col_validity_q{N}":
        return "row_col_val_qN"
    return result_name


def _add_certificate_columns(row: dict[str, object], metric_name: str, result: CertificateResult) -> None:
    row[metric_name] = "" if result.B_star is None else result.B_star
    row[f"{metric_name}_status"] = result.status_name
    row[f"{metric_name}_is_optimal"] = result.is_optimal
    row[f"{metric_name}_lower_bound"] = "" if result.lower_bound is None else result.lower_bound
    row[f"{metric_name}_upper_bound"] = "" if result.upper_bound is None else result.upper_bound
    row[f"{metric_name}_mip_gap"] = "" if result.mip_gap is None else result.mip_gap


def _fill_degenerate_corner_columns(row: dict[str, object]) -> None:
    if int(row["L"]) == 1:
        if "row_col_stab_q1_r1" in row:
            _copy_metric_family(row, "row_col_stab_q1_r1", "row_col_stab_q1_rL")
        if "row_col_stab_qN_r1" in row:
            _copy_metric_family(row, "row_col_stab_qN_r1", "row_col_stab_qN_rL")
    if int(row["N"]) == 1:
        if "row_col_stab_q1_r1" in row:
            _copy_metric_family(row, "row_col_stab_q1_r1", "row_col_stab_qN_r1")
        if "row_col_stab_q1_rL" in row:
            _copy_metric_family(row, "row_col_stab_q1_rL", "row_col_stab_qN_rL")
        if "row_col_val_q1" in row:
            _copy_metric_family(row, "row_col_val_q1", "row_col_val_qN")


def _copy_metric_family(row: dict[str, object], source: str, target: str) -> None:
    row.setdefault(target, row[source])
    for suffix in ["_status", "_is_optimal", "_lower_bound", "_upper_bound", "_mip_gap"]:
        source_key = f"{source}{suffix}"
        if source_key in row:
            row.setdefault(f"{target}{suffix}", row[source_key])


def _add_validity_demo_gap_columns(row: dict[str, object]) -> None:
    """Add validity-demo comparison columns when the needed values are present."""
    joint = _numeric_value(row.get("row_col_val_q1"))
    tpa = _numeric_value(row.get("tpa_val_sequence_q1"))
    row["validity_gap_joint_minus_tpa_q1"] = _safe_difference(joint, tpa)
    row["validity_ratio_joint_over_tpa_q1"] = _safe_ratio(joint, tpa)
    plain_dpa = _numeric_value(row.get("plain_dpa_val_sequence_q1"))
    row["validity_gap_tpa_minus_plain_dpa_q1"] = _safe_difference(tpa, plain_dpa)
    row["validity_gap_joint_minus_plain_dpa_q1"] = _safe_difference(joint, plain_dpa)


def _validity_demo_diagnostics(data: ToyData) -> dict[str, object]:
    """Summarize generator-local validity diagnostics for CSV/audit output."""
    L = data.val_votes.shape[2]
    plain_dpa_cells = _plain_dpa_validity_token_budgets(data)
    dpa_cells = _cell_validity_budgets(data)
    tpa_cells = _targeted_validity_token_budgets(data)
    plain_phrase = plain_dpa_cells.max(axis=1)
    tpa_phrase = tpa_cells.max(axis=1)
    groups = [_influenced_group_for_token(data, j) for j in range(L)]
    union_size = len(set().union(*(set(group) for group in groups))) if groups else 0
    example_positions = range(min(3, L))
    return {
        "validity_demo_vote_counts_row0": ";".join(",".join(str(int(value)) for value in data.val_counts[0, j]) for j in range(L)),
        "validity_demo_vote_count_examples_row0": ";".join(
            f"j={j}:" + ",".join(str(int(value)) for value in data.val_counts[0, j]) for j in example_positions
        ),
        "validity_demo_plain_dpa_token_radii_row0": ",".join(str(int(value)) for value in plain_dpa_cells[0]),
        "validity_demo_dpa_token_radii_row0": ",".join(str(int(value)) for value in dpa_cells[0]),
        "validity_demo_tpa_token_radii_row0": ",".join(str(int(value)) for value in tpa_cells[0]),
        "validity_demo_tpa_phrase_radius_row0": int(np.max(tpa_cells[0])),
        "validity_demo_plain_phrase_radii_summary": _summary_string(plain_phrase),
        "validity_demo_tpa_phrase_radii_summary": _summary_string(tpa_phrase),
        "validity_demo_plain_token_radii_summary": _summary_string(plain_dpa_cells.reshape(-1)),
        "validity_demo_tpa_token_radii_summary": _summary_string(tpa_cells.reshape(-1)),
        "validity_demo_pct_tpa_gt_plain": float(np.mean(tpa_cells > plain_dpa_cells) * 100.0),
        "validity_demo_pct_tpa_eq_plain": float(np.mean(tpa_cells == plain_dpa_cells) * 100.0),
        "validity_demo_pct_tpa_lt_plain": float(np.mean(tpa_cells < plain_dpa_cells) * 100.0),
        "validity_demo_shard_groups_row0": ";".join(",".join(str(k) for k in group) for group in groups),
        "validity_demo_required_group_union_size": union_size,
        "validity_demo_group_feasible_all": True,
    }


def _summary_string(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "none"
    return (
        f"min={np.min(finite):.6g},p25={np.percentile(finite, 25):.6g},"
        f"median={np.median(finite):.6g},mean={np.mean(finite):.6g},"
        f"p75={np.percentile(finite, 75):.6g},max={np.max(finite):.6g}"
    )


def _influenced_group_for_token(data: ToyData, j: int, row: int = 0) -> list[int]:
    return [int(k) for k in np.flatnonzero(data.influence[:, row, j])]


def plot_benchmark_csv(csv_path: str, save_dir: str | None = None) -> list[dict[str, object]]:
    """Read a benchmark CSV and regenerate the default report-facing plots."""
    path = Path(csv_path)
    output_dir = Path(save_dir) if save_dir is not None else path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows_csv(path)
    save_default_report_plots(rows, output_dir, csv_path=path)
    print(f"Wrote benchmark plots under: {output_dir}")
    return rows


SWEEP_STABILITY_METRICS = {
    "DPA weakest-token stability": "dpa_stab_row_radius_qN",
    "Shared MILP, one token per row": "row_col_stab_qN_r1",
    "Shared MILP, full token grid": "row_col_stab_qN_rL",
}

SWEEP_VALIDITY_METRICS = {
    "Plain DPA validity": "plain_dpa_val_sequence_qN",
    "TPA max-token phrase baseline": "tpa_val_sequence_qN",
    "Shared MILP validity": "row_col_val_qN",
}


def plot_sweep_csv(csv_path: str, sweep: str, save_dir: str | None = None) -> list[dict[str, object]]:
    """Render a synthetic scaling sweep from an existing benchmark CSV."""
    path = Path(csv_path)
    output_dir = Path(save_dir) if save_dir is not None else path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows_csv(path)
    if not rows:
        raise SystemExit(f"Sweep CSV is empty: {path}")

    sweep_key = sweep.upper() if sweep.lower() != "degenerate" else "degenerate"
    if sweep_key not in {"K", "N", "L", "degenerate"}:
        raise SystemExit(f"Unknown sweep {sweep!r}; expected K, N, L, or degenerate")

    seed_values = sorted(
        {
            int(seed)
            for row in rows
            if (seed := _numeric_value(row.get("seed"))) is not None
        }
    )
    plot_rows = rows
    if len(seed_values) > 1:
        aggregate_path = path.parent / "benchmark_results_seed_aggregate.csv"
        plot_rows = _aggregate_result_rows_across_seeds(rows)
        _write_rows_csv(aggregate_path, plot_rows)
        budget_path = path.parent / "benchmark_budget_curves.csv"
        if budget_path.exists():
            aggregate_budget_path = path.parent / "benchmark_budget_curves_seed_aggregate.csv"
            _write_rows_csv(
                aggregate_budget_path,
                _aggregate_budget_rows_across_seeds(_read_rows_csv(budget_path)),
            )

    generated: list[str] = []
    skipped: list[str] = []
    if sweep_key == "degenerate":
        compact_csv_path = output_dir / "degenerate_study.csv"
        table_path = output_dir / "degenerate_study_table.tex"
        _write_degenerate_sweep_csv(compact_csv_path, plot_rows)
        _write_degenerate_sweep_table(table_path, plot_rows)
        generated.extend([compact_csv_path.name, table_path.name])
    else:
        metric_groups = [
            ("stability", SWEEP_STABILITY_METRICS),
            ("validity", SWEEP_VALIDITY_METRICS),
            ("runtime", {"Total Gurobi objective runtime": "runtime_gurobi_total"}),
        ]
        for plot_kind, metrics in metric_groups:
            filename = f"sweep_{sweep_key}_{plot_kind}_certificate_vs_{sweep_key}.pdf"
            y_label = "Mean certified budget B*"
            if plot_kind == "runtime":
                filename = f"sweep_{sweep_key}_runtime_vs_{sweep_key}.pdf"
                y_label = "Mean Gurobi runtime (seconds)"
            series, ranges, metric_skips = _metric_series_with_ranges(plot_rows, sweep_key, metrics)
            skipped.extend(f"{filename}: {message}" for message in metric_skips)
            if not series:
                skipped.append(f"{filename}: no numeric series available")
                continue
            _save_line_plot(
                output_dir / filename,
                f"{plot_kind.capitalize()} scaling with {sweep_key}",
                _axis_label(sweep_key),
                y_label,
                series,
                ranges=ranges,
            )
            generated.append(filename)

    _write_sweep_audit(
        output_dir / "audit_sweep.md",
        rows=rows,
        csv_path=path,
        sweep=sweep_key,
        generated=generated,
        skipped=skipped,
    )
    print(f"Wrote {sweep_key} sweep plots under: {output_dir}")
    return rows


def _write_degenerate_sweep_table(path: Path, rows: list[dict[str, object]]) -> None:
    metrics = _degenerate_sweep_metrics()
    lines = [
        r"\begin{tabular}{rr" + "r" * len(metrics) + "}",
        r"\toprule",
        "N & L & " + " & ".join(label for label, _metric in metrics) + r" \\",
        r"\midrule",
    ]
    for row in sorted(rows, key=lambda item: (float(item["N"]), float(item["L"]))):
        values = [_format_table_number(row.get(metric)) for _label, metric in metrics]
        lines.append(f"{row['N']} & {row['L']} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def _write_degenerate_sweep_csv(path: Path, rows: list[dict[str, object]]) -> None:
    metrics = _degenerate_sweep_metrics()
    compact_rows = []
    for row in sorted(rows, key=lambda item: (float(item["N"]), float(item["L"]))):
        compact_row = {"N": row.get("N"), "L": row.get("L")}
        compact_row.update({metric: row.get(metric, "") for _label, metric in metrics})
        compact_rows.append(compact_row)
    _write_rows_csv(path, compact_rows)


def _degenerate_sweep_metrics() -> list[tuple[str, str]]:
    return [
        ("DPA stability", "dpa_stab_row_radius_qN"),
        ("Shared stability, r=1", "row_col_stab_qN_r1"),
        ("Shared stability, r=L", "row_col_stab_qN_rL"),
        ("Plain DPA validity", "plain_dpa_val_sequence_qN"),
        ("TPA max-token phrase baseline", "tpa_val_sequence_qN"),
        ("Shared validity", "row_col_val_qN"),
    ]


def _format_table_number(value: object) -> str:
    numeric = _numeric_value(value)
    if numeric is None:
        return "--"
    return f"{numeric:.3f}".rstrip("0").rstrip(".")


def _write_sweep_audit(
    path: Path,
    *,
    rows: list[dict[str, object]],
    csv_path: Path,
    sweep: str,
    generated: list[str],
    skipped: list[str],
) -> None:
    expected_varied = {"degenerate": {"N", "L"}}.get(sweep, {sweep})
    allowed_varied = expected_varied | {"seed"}
    parameter_keys = ["K", "N", "L", "T", "delta_stab", "delta_val", "target_bias", "seed"]
    unique_values = {key: _sorted_unique_values(rows, key) for key in parameter_keys}
    unexpectedly_varied = [
        key for key, values in unique_values.items() if len(values) > 1 and key not in allowed_varied
    ]
    expected_but_fixed = [key for key in expected_varied if len(unique_values.get(key, [])) < 2]
    status_columns = sorted({key for row in rows for key in row if key.endswith("_status")})
    requested_metrics = {**SWEEP_STABILITY_METRICS, **SWEEP_VALIDITY_METRICS}
    requested_metrics["Total Gurobi objective runtime"] = "runtime_gurobi_total"
    included_methods = [
        f"{label} (`{metric}`)"
        for label, metric in requested_metrics.items()
        if any(_numeric_value(row.get(metric)) is not None for row in rows)
    ]
    missing_metrics = [
        f"{label} (`{metric}`)"
        for label, metric in requested_metrics.items()
        if not any(_numeric_value(row.get(metric)) is not None for row in rows)
    ]

    lines = [
        "# Sweep benchmark audit",
        "",
        f"- Source CSV: `{csv_path}`",
        f"- Sweep: `{sweep}`",
        f"- Rows: {len(rows)}",
        f"- Expected varied parameter(s): {', '.join(sorted(expected_varied))}",
        f"- Fixed vocabulary size `T`: {'yes' if len(unique_values['T']) == 1 else 'NO'}",
        "",
        "## Parameter values",
        "",
    ]
    lines.extend(f"- `{key}`: {values}" for key, values in unique_values.items())
    lines.extend(["", "## Generated artifacts", ""])
    lines.extend(f"- `{filename}`" for filename in generated)
    if not generated:
        lines.append("- None")
    lines.extend(["", "## Methods and metrics", ""])
    lines.append(f"- Included: {included_methods or 'none'}")
    lines.append(f"- Missing or non-numeric: {missing_metrics or 'none'}")
    lines.extend(["", "## Solver statuses", ""])
    if status_columns:
        for column in status_columns:
            values = sorted({str(row.get(column, "")) for row in rows if row.get(column, "") != ""})
            lines.append(f"- `{column}`: {values or ['missing']}")
    else:
        lines.append("- No `*_status` columns found.")
    lines.extend(["", "## Checks", ""])
    lines.append(f"- Unexpectedly varied parameters: {unexpectedly_varied or 'none'}")
    lines.append(f"- Expected varied parameters with fewer than two values: {expected_but_fixed or 'none'}")
    lines.append(f"- Skipped series or plots: {skipped or 'none'}")
    path.write_text("\n".join(lines) + "\n")


RELATIVE_LIFT_PERCENT_REPORTING_THRESHOLD = 1.0

def save_default_report_plots(rows: list[dict[str, object]], output_dir: Path, csv_path: Path) -> None:
    """Write only the simplified default report plot set."""
    _clean_default_plot_dir(output_dir)
    budget_rows = _read_optional_csv(csv_path.parent / "benchmark_budget_curves.csv")
    audit: list[dict[str, object]] = []

    stability_series, stability_skipped = _budget_curve_series(
        budget_rows,
        [
            ("DPA weakest token", "DPA token margin", "full_response_stable_against_any_token_change", "radius_derived"),
            ("Shared MILP full sequence", "Shared MILP", "stability_full_sequence_per_prompt", "radius_derived"),
            ("Shared MILP one token per row", "Shared MILP", "stability_one_token_per_prompt", "radius_derived"),
        ],
    )
    if stability_series:
        _save_line_plot(
            output_dir / "stability_budget_curve.pdf",
            "Stability budget curve",
            "Poisoned shard budget B",
            "Certified fraction (%)",
            stability_series,
        )
    else:
        print("Warning: skipped stability_budget_curve.pdf; no requested budget-curve series were available.")
    audit.append(
        {
            "plot": "stability_budget_curve.pdf",
            "series": list(stability_series),
            "series_data": stability_series,
            "comparisons": _performance_comparison_lines(
                stability_series,
                [
                    (
                        "Shared MILP full sequence",
                        "DPA weakest token",
                        "average certified-fraction lift",
                        "percentage points",
                    ),
                    (
                        "Shared MILP one token per row",
                        "DPA weakest token",
                        "average certified-fraction lift",
                        "percentage points",
                    )
                ],
            ),
            "skipped": stability_skipped,
        }
    )

    validity_series, validity_skipped = _budget_curve_series(
        budget_rows,
        [
            ("Shared shard-aware MILP full sequence", "Shared MILP", "validity_full_harmful_sequence_per_prompt", "radius_derived"),
            ("TPA max-token phrase baseline", "TPA max-token phrase baseline", "validity_full_harmful_sequence_per_prompt", "radius_derived"),
            ("Plain DPA max-token phrase blocker", "Plain DPA max-token phrase blocker", "validity_full_harmful_sequence_per_prompt", "radius_derived"),
        ],
    )
    if validity_series:
        _save_line_plot(output_dir / "validity_budget_curve.pdf", "Validity budget curve", "Poisoned shard budget B", "Certified fraction (%)", validity_series)
    else:
        print("Warning: skipped validity_budget_curve.pdf; no requested budget-curve series were available.")
    audit.append(
        {
            "plot": "validity_budget_curve.pdf",
            "series": list(validity_series),
            "series_data": validity_series,
            "comparisons": _performance_comparison_lines(
                validity_series,
                [
                    (
                        "Shared shard-aware MILP full sequence",
                        "TPA max-token phrase baseline",
                        "average certified-fraction lift",
                        "percentage points",
                    ),
                    (
                        "TPA max-token phrase baseline",
                        "Plain DPA max-token phrase blocker",
                        "average certified-fraction lift",
                        "percentage points",
                    ),
                    (
                        "Shared shard-aware MILP full sequence",
                        "Plain DPA max-token phrase blocker",
                        "average certified-fraction lift",
                        "percentage points",
                    ),
                ],
            ),
            "skipped": validity_skipped,
        }
    )

    metric_specs = _default_report_metric_specs(csv_path)
    for filename, title, metrics in metric_specs:
        series, skipped = _metric_series(rows, "K", metrics)
        if series:
            _save_line_plot(output_dir / filename, title, _axis_label("K"), "Mean certified budget B*", series)
        else:
            print(f"Warning: skipped {filename}; no requested benchmark-result series were available.")
        audit.append(
            {
                "plot": filename,
                "series": list(series),
                "series_data": series,
                "comparisons": _metric_plot_comparison_lines(filename, series),
                "skipped": skipped,
            }
        )

    _write_default_plot_audit(output_dir / "audit_plot_outputs.txt", csv_path, audit)
    _write_comparison_report(output_dir / "comparisons.txt", csv_path, audit, rows=rows)


def _default_report_metric_specs(csv_path: Path) -> list[tuple[str, str, dict[str, str]]]:
    """Keep the standard size reports focused on budget curves."""
    if _is_standard_size_benchmark_csv(csv_path):
        return []
    return [
        (
            "stability_certificate_vs_K.pdf",
            "Stability certificate vs K",
            MAIN_STABILITY_METRICS,
        ),
        (
            "validity_certificate_vs_K.pdf",
            "Validity certificate vs K",
            MAIN_VALIDITY_METRICS,
        ),
    ]


def _is_standard_size_benchmark_csv(csv_path: Path) -> bool:
    """Limit the summative validity comparison to small/medium/large presets."""
    return (
        csv_path.name == "benchmark_results.csv"
        and csv_path.parent.name == "results"
        and csv_path.parent.parent.name in {"small", "medium", "large"}
    )


def _clean_default_plot_dir(output_dir: Path) -> None:
    """Remove old generated plot artifacts from the default plot directory."""
    for child in output_dir.iterdir():
        if child.is_file() and (child.suffix == ".pdf" or child.name.startswith("audit_")):
            child.unlink()


def _budget_curve_series(
    rows: list[dict[str, object]],
    selections: list[tuple[str, str, str, str]],
) -> tuple[dict[str, tuple[list[float], list[float]]], list[str]]:
    series: dict[str, tuple[list[float], list[float]]] = {}
    skipped: list[str] = []
    if not rows:
        return series, ["benchmark_budget_curves.csv missing or empty"]
    for label, method, objective, curve_type in selections:
        selected_rows = [
            row
            for row in rows
            if row.get("method") == method and row.get("objective") == objective and row.get("curve_type") == curve_type
        ]
        metric_series, diagnostic = _fixed_denominator_budget_series(selected_rows)
        if metric_series is None:
            skipped.append(f"{label}: {diagnostic or f'missing method={method}, objective={objective}, curve_type={curve_type}'}")
            continue
        if not _is_nonincreasing(metric_series[1]):
            skipped.append(f"{label}: warning non-monotonic certified fraction after fixed-denominator aggregation")
        xs, ys = metric_series
        series[label] = (xs, [100.0 * y for y in ys])
    return series, skipped


def _fixed_denominator_budget_series(rows: list[dict[str, object]]) -> tuple[tuple[list[float], list[float]] | None, str | None]:
    if not rows:
        return None, "no matching budget rows"
    grouped: dict[tuple[tuple[str, object], ...], dict[float, float]] = {}
    for row in rows:
        budget = _numeric_value(row.get("budget"))
        value = _numeric_value(row.get("certified_fraction"))
        if budget is None or value is None:
            continue
        key = _budget_config_key(row)
        grouped.setdefault(key, {})[budget] = value
    if not grouped:
        return None, "no numeric budget rows"
    budget_sets = [set(values) for values in grouped.values()]
    common_budgets = sorted(set.intersection(*budget_sets))
    if not common_budgets:
        return None, "no common budget range across configurations"
    nonmonotonic_groups = 0
    for values in grouped.values():
        ys = [values[budget] for budget in common_budgets]
        if not _is_nonincreasing(ys):
            nonmonotonic_groups += 1
    ys = [float(np.mean([values[budget] for values in grouped.values()])) for budget in common_budgets]
    diagnostic = None
    if nonmonotonic_groups:
        diagnostic = f"{nonmonotonic_groups} config group(s) are non-monotonic"
    return (common_budgets, ys), diagnostic


def _budget_config_key(row: dict[str, object]) -> tuple[tuple[str, object], ...]:
    ignored = {
        "budget",
        "certified_fraction",
        "attacked_fraction",
        "mean_radius",
        "median_radius",
        "min_radius",
        "max_radius",
        "num_certified",
        "num_known",
        "num_unknown",
        "num_total",
    }
    return tuple(sorted((key, value) for key, value in row.items() if key not in ignored))


def _is_nonincreasing(values: list[float]) -> bool:
    return all(right <= left + 1e-12 for left, right in zip(values, values[1:]))


def _metric_series(
    rows: list[dict[str, object]],
    axis_name: str,
    metrics: dict[str, str],
) -> tuple[dict[str, tuple[list[float], list[float]]], list[str]]:
    series: dict[str, tuple[list[float], list[float]]] = {}
    skipped: list[str] = []
    for label, metric in metrics.items():
        metric_rows = rows
        if metric in SENTINEL_VALIDITY_METRICS:
            metric_rows = [row for row in rows if not _is_sentinel_budget(row.get(metric), row.get("K"))]
        metric_series = _mean_series_by_axis(metric_rows, axis_name, metric)
        if metric_series is None:
            skipped.append(f"{label}: missing or empty column {metric}")
            continue
        series[label] = metric_series
    return series, skipped


def _metric_series_with_ranges(
    rows: list[dict[str, object]],
    axis_name: str,
    metrics: dict[str, str],
) -> tuple[
    dict[str, tuple[list[float], list[float]]],
    dict[str, tuple[list[float], list[float], list[float]]],
    list[str],
]:
    series, skipped = _metric_series(rows, axis_name, metrics)
    ranges: dict[str, tuple[list[float], list[float], list[float]]] = {}
    for label, metric in metrics.items():
        if label not in series:
            continue
        lower = _mean_series_by_axis(rows, axis_name, f"{metric}_min")
        upper = _mean_series_by_axis(rows, axis_name, f"{metric}_max")
        if lower is None or upper is None or lower[0] != upper[0]:
            continue
        ranges[label] = (lower[0], lower[1], upper[1])
    return series, ranges, skipped


SENTINEL_VALIDITY_METRICS = {
    "dpa_val_cell_min",
    "dpa_val_row_weak_q1",
    "dpa_val_row_weak_qN",
    "dpa_val_row_strong_q1",
    "dpa_val_row_strong_qN",
    "raw_dpa_val_min_cell",
}


def _is_sentinel_budget(value: object, K: object) -> bool:
    numeric_value = _numeric_value(value)
    numeric_k = _numeric_value(K)
    return numeric_value is not None and numeric_k is not None and numeric_value > numeric_k


def _metric_plot_comparison_lines(filename: str, series: dict[str, tuple[list[float], list[float]]]) -> list[str]:
    if filename == "stability_certificate_vs_K.pdf":
        return _performance_comparison_lines(
            series,
            [("Shared MILP full matrix", "DPA weakest token", "mean certified-budget lift", "budget units")],
        )
    if filename == "validity_certificate_vs_K.pdf":
        return _performance_comparison_lines(
            series,
            [
                (
                    "Shared shard-aware MILP full sequence",
                    "TPA max-token phrase baseline",
                    "mean certified-budget lift",
                    "budget units",
                ),
                (
                    "TPA max-token phrase baseline",
                    "Plain DPA max-token phrase blocker",
                    "mean certified-budget lift",
                    "budget units",
                ),
                (
                    "Shared shard-aware MILP full sequence",
                    "Plain DPA max-token phrase blocker",
                    "mean certified-budget lift",
                    "budget units",
                ),
            ],
        )
    return []


def _performance_comparison_lines(
    series: dict[str, tuple[list[float], list[float]]],
    comparisons: list[tuple[str, str, str, str]],
) -> list[str]:
    lines = []
    for better_label, baseline_label, quantity_label, unit_label in comparisons:
        comparison = _series_relative_improvement(series, better_label, baseline_label)
        if comparison is None:
            lines.append(f"{better_label} vs {baseline_label}: unavailable")
            continue
        lift, percent = comparison
        lines.append(
            f"{better_label} vs {baseline_label}: {quantity_label} {lift:.3g} {unit_label}; relative lift {percent:.3g}%"
        )
    return lines


def _series_relative_improvement(
    series: dict[str, tuple[list[float], list[float]]],
    better_label: str,
    baseline_label: str,
) -> tuple[float, float] | None:
    if better_label not in series or baseline_label not in series:
        return None
    left_xs, left_ys = series[better_label]
    right_xs, right_ys = series[baseline_label]
    left_by_x = {x: y for x, y in zip(left_xs, left_ys)}
    right_by_x = {x: y for x, y in zip(right_xs, right_ys)}
    common_xs = sorted(set(left_by_x) & set(right_by_x))
    if not common_xs:
        return None
    left_mean = float(np.mean([left_by_x[x] for x in common_xs]))
    right_mean = float(np.mean([right_by_x[x] for x in common_xs]))
    lift = max(0.0, left_mean - right_mean)
    return lift, _relative_lift_percent(lift, right_mean)


def _write_default_plot_audit(path: Path, csv_path: Path, audit: list[dict[str, object]]) -> None:
    generated = [item["plot"] for item in audit if (path.parent / str(item["plot"])).exists()]
    lines = [
        "Default report plot audit",
        "",
        f"CSV file used: {csv_path}",
        "plot.sh reruns Gurobi: no",
        "",
        "Certified fraction definition:",
        "certified_fraction(B) = 100 * mean[ B < B_star ]",
        "B_star is the minimum attack budget returned by the corresponding baseline or MILP certificate.",
        "If B equals B_star, the attack is feasible, so the region is not certified at that budget.",
        "For stability, this is the percentage of evaluated stability regions where the clean output is guaranteed not to change under budget B.",
        "For validity, this is the percentage of evaluated validity regions where the harmful target output is guaranteed not to be forceable under budget B.",
        "",
        "Plots generated:",
        *[f"- {name}" for name in generated],
        "",
        "Plot series:",
    ]
    for item in audit:
        lines.append(f"- {item['plot']}: {', '.join(item['series']) if item['series'] else 'none'}")
        comparisons = item.get("comparisons") or []
        for comparison in comparisons:
            lines.append(f"  comparison: {comparison}")
        skipped = item["skipped"]
        if skipped:
            lines.append(f"  skipped: {'; '.join(skipped)}")
    path.write_text("\n".join(lines))


def _write_comparison_report(
    path: Path,
    csv_path: Path,
    audit: list[dict[str, object]],
    *,
    rows: list[dict[str, object]],
) -> None:
    lines = [
        "Toy experiment strategy comparisons",
        "",
        f"CSV file used: {csv_path}",
        "Generated by plotting only: yes",
        "Reruns Gurobi or regenerates experiments: no",
        "",
        "Math:",
        "absolute lift = max(0, mean(strategy A) - mean(strategy B))",
        "relative lift = 100 * absolute_lift / abs(mean(strategy B))",
        f"relative lifts below {RELATIVE_LIFT_PERCENT_REPORTING_THRESHOLD:g}% are reported as 0%",
        "",
        "Expected ordering checks:",
        f"- Stability full-sequence MILP > DPA observed in budget curves: {_comparison_is_positive(audit, 'stability_budget_curve.pdf', 'Shared MILP full sequence', 'DPA weakest token')}",
        f"- Stability one-token MILP > DPA observed in budget curves: {_comparison_is_positive(audit, 'stability_budget_curve.pdf', 'Shared MILP one token per row', 'DPA weakest token')}",
        f"- Validity MILP > TPA observed in budget curves: {_comparison_is_positive(audit, 'validity_budget_curve.pdf', 'Shared shard-aware MILP full sequence', 'TPA max-token phrase baseline')}",
        f"- Validity TPA > DPA observed in budget curves: {_comparison_is_positive(audit, 'validity_budget_curve.pdf', 'TPA max-token phrase baseline', 'Plain DPA max-token phrase blocker')}",
        "",
        "Comparison summary:",
    ]
    for item in audit:
        comparisons = item.get("comparisons") or []
        if not comparisons:
            continue
        lines.append(f"- {item['plot']}")
        for comparison in comparisons:
            lines.append(f"  - {comparison}")
    if _is_standard_size_benchmark_csv(csv_path):
        lines.extend(["", "Summative validity composition:"])
        lines.extend(f"- {line}" for line in _summative_validity_stat_lines(rows))
    lines.extend(["", "Series summary stats:"])
    for item in audit:
        series_data = item.get("series_data") or {}
        if not series_data:
            continue
        lines.append(f"- {item['plot']}")
        lines.extend(f"  - {line}" for line in _series_summary_stat_lines(series_data))
    path.write_text("\n".join(lines))


def _summative_validity_stat_lines(rows: list[dict[str, object]]) -> list[str]:
    """Compare additive independent validity costs with shared-shard MILP costs."""
    series, _skipped = _metric_series(
        rows,
        "K",
        {
            "Summative DPA validity": "independent_val_sequence_qN",
            "Shared MILP validity": "row_col_val_qN",
        },
    )
    if len(series) != 2:
        return ["unavailable: required validity metrics are missing"]
    dpa_xs, dpa_ys = series["Summative DPA validity"]
    milp_xs, milp_ys = series["Shared MILP validity"]
    dpa_by_k = dict(zip(dpa_xs, dpa_ys))
    milp_by_k = dict(zip(milp_xs, milp_ys))
    common_ks = sorted(set(dpa_by_k) & set(milp_by_k))
    if not common_ks:
        return ["unavailable: no common K values"]
    dpa_mean = float(np.mean([dpa_by_k[K] for K in common_ks]))
    milp_mean = float(np.mean([milp_by_k[K] for K in common_ks]))
    overestimate = dpa_mean - milp_mean
    ratio = dpa_mean / milp_mean if milp_mean != 0 else float("inf")
    lines = [
        "Definition: summative DPA is `independent_val_sequence_qN`; shared MILP is `row_col_val_qN`.",
        f"Overall mean summative DPA budget: {dpa_mean:.6g}",
        f"Overall mean shared MILP budget: {milp_mean:.6g}",
        f"Mean additive overestimate: {overestimate:.6g} budget units",
        f"Mean summative-DPA/shared-MILP ratio: {ratio:.6g}x",
    ]
    lines.extend(
        f"K={K:g}: summative DPA={dpa_by_k[K]:.6g}, shared MILP={milp_by_k[K]:.6g}, "
        f"difference={dpa_by_k[K] - milp_by_k[K]:.6g}"
        for K in common_ks
    )
    return lines


def _comparison_is_positive(
    audit: list[dict[str, object]],
    plot_name: str,
    left_label: str,
    right_label: str,
) -> bool:
    for item in audit:
        if item.get("plot") != plot_name:
            continue
        comparison = _series_relative_improvement(item.get("series_data") or {}, left_label, right_label)
        return bool(comparison is not None and comparison[0] > 1e-12)
    return False


def _series_summary_stat_lines(series: dict[str, tuple[list[float], list[float]]]) -> list[str]:
    lines = []
    for label, (xs, ys) in series.items():
        if not ys:
            continue
        values = np.asarray(ys, dtype=float)
        lines.append(
            f"{label}: n={len(ys)}, x={min(xs):.6g}..{max(xs):.6g}, "
            f"mean={np.mean(values):.6g}, min={np.min(values):.6g}, max={np.max(values):.6g}"
        )
    return lines or ["no numeric series"]


def _safe_difference(left: float | None, right: float | None) -> float | str:
    if left is None or right is None:
        return ""
    return float(left - right)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | str:
    if numerator is None or denominator is None or denominator == 0:
        return ""
    return float(numerator / denominator)


def _sorted_unique_values(rows: list[dict[str, object]], key: str) -> list[object]:
    return sorted({row.get(key) for row in rows if row.get(key) not in {None, ""}})


def _format_mean_series(rows: list[dict[str, object]], metric: str) -> list[str]:
    series = _mean_series_by_axis(rows, "L", metric)
    if series is None:
        return ["- no numeric rows"]
    xs, ys = series
    return [f"- L={x:g}: {y:.6g}" for x, y in zip(xs, ys)]


def print_console_grid(data: ToyData) -> None:
    """Print a compact per-cell grid and shard vote layers."""
    margins = stability_margins(data.stab_counts, data.clean_pred)
    winner_counts = np.take_along_axis(data.stab_counts, data.clean_pred[:, :, None], axis=2)[:, :, 0]
    target_counts = np.take_along_axis(data.val_counts, data.target[:, :, None], axis=2)[:, :, 0]

    print()
    print("cell grid: pred->target | runner | stab_margin | stab_winner_count | val_target_count")
    for i in range(data.clean_pred.shape[0]):
        cells = []
        for j in range(data.clean_pred.shape[1]):
            cells.append(
                f"{int(data.clean_pred[i, j])}->{int(data.target[i, j])}"
                f" r{int(data.runner_up[i, j])}"
                f" m{int(margins[i, j])}"
                f" w{int(winner_counts[i, j])}"
                f" h{int(target_counts[i, j])}"
            )
        print(f"row {i:02d}: " + " | ".join(cells))

    max_layers_to_print = 20
    print()
    print("stability shard vote layers stab_votes[k, row, col]:")
    for k in range(min(data.votes.shape[0], max_layers_to_print)):
        print(f"k={k:02d}")
        print(data.stab_votes[k])
    if data.votes.shape[0] > max_layers_to_print:
        print(f"... omitted {data.votes.shape[0] - max_layers_to_print} shard layers")

    print()
    print("validity shard vote layers val_votes[k, row, col]:")
    for k in range(min(data.val_votes.shape[0], max_layers_to_print)):
        print(f"k={k:02d}")
        print(data.val_votes[k])
    if data.val_votes.shape[0] > max_layers_to_print:
        print(f"... omitted {data.val_votes.shape[0] - max_layers_to_print} shard layers")


def save_instance_plots(
    data: ToyData,
    output_dir: Path,
    K: int,
    N: int,
    L: int,
    T: int,
    delta_stab: float,
    delta_val: float,
    target_bias: float,
    seed: int,
    influence_mode: str,
) -> None:
    """Write per-instance heatmaps and curves for one generated toy instance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    margins = stability_margins(data.stab_counts, data.clean_pred)
    target_counts = np.take_along_axis(data.val_counts, data.target[:, :, None], axis=2)[:, :, 0]
    stability_grid = compute_structured_stability_grid(data)
    q_curve = compute_validity_q_curve(data, T)

    title_suffix = (
        f"K={K}, N={N}, L={L}, T={T}, delta_stab={delta_stab}, delta_val={delta_val}, "
        f"target_bias={target_bias}, influence={influence_mode}, seed={seed}"
    )
    _save_heatmap_pdf(data.clean_pred, output_dir / "clean_predictions.pdf", "Clean predictions | " + title_suffix)
    _save_heatmap_pdf(data.target, output_dir / "harmful_targets.pdf", "Harmful targets | " + title_suffix)
    _save_heatmap_pdf(margins, output_dir / "stability_margins.pdf", "Winner vs runner-up margins | " + title_suffix)
    _save_heatmap_pdf(target_counts, output_dir / "validity_target_counts.pdf", "Target validity vote counts | " + title_suffix)
    _save_heatmap_pdf(
        stability_grid,
        output_dir / "structured_stability_heatmap.pdf",
        "Structured stability poison budget | " + title_suffix,
        x_label="affected tokens per prompt",
        y_label="affected prompts",
        colorbar_label="poison budget B*",
    )
    _save_line_plot_pdf(
        output_dir / "validity_q_curve.pdf",
        "Row-column validity B*(q) | " + title_suffix,
        "q rows compromised",
        "Poison budget B*",
        {"shared MILP": ([float(q) for q in range(1, N + 1)], [float(v) for v in q_curve])},
    )

    print()
    print(f"Wrote instance plots under: {output_dir}")


CANONICAL_COLORS = {
    "Shared MILP one prompt, one token": "#1f77b4",
    "Shared MILP one prompt, full sequence": "#17becf",
    "Shared MILP all prompts, one token each": "#9467bd",
    "Shared MILP full matrix": "#08519c",
    "Shared MILP full sequence": "#1f77b4",
    "Shared shard-aware MILP full sequence": "#1f77b4",
    "Shared MILP all harmful sequences": "#08519c",
    "DPA weakest token": "#d62728",
    "Plain DPA count-margin phrase blocker": "#d62728",
    "Plain DPA max-token phrase blocker": "#d62728",
    "DPA weakest harmful token": "#8c564b",
    "DPA most difficult harmful token": "#e377c2",
    "TPA max-token sequence": "#2ca02c",
    "TPA max-token phrase baseline": "#2ca02c",
    "Shard-aware weakest-token diagnostic": "#8c564b",
    "Shard-aware independent max-token diagnostic": "#e377c2",
    "Independent shard-aware composition diagnostic": "#7f7f7f",
}

MAIN_STABILITY_METRICS = {
    "Shared MILP full matrix": "row_col_stab_qN_rL",
    "DPA weakest token": "dpa_stab_row_radius_qN",
}

MAIN_VALIDITY_METRICS = {
    "Shared shard-aware MILP full sequence": "row_col_val_qN",
    "TPA max-token phrase baseline": "tpa_val_sequence_qN",
    "Plain DPA max-token phrase blocker": "plain_dpa_val_sequence_qN",
}

def save_validity_demo_plot(rows: list[dict[str, object]], output_dir: Path, generator: str) -> None:
    """Write the controlled validity demo plot when matching rows are present."""
    demo_rows = [row for row in rows if row.get("generator") == generator]
    if not demo_rows:
        return
    series_specs = [
        ("Shared shard-aware MILP full sequence", "row_col_val_q1"),
        ("TPA max-token phrase baseline", "tpa_val_sequence_q1"),
        ("Plain DPA max-token phrase blocker", "plain_dpa_val_sequence_q1"),
    ]
    series: dict[str, tuple[list[float], list[float]]] = {}
    skipped: list[str] = []
    for label, column in series_specs:
        metric_series = _mean_series_by_axis(demo_rows, "L", column)
        if metric_series is None:
            skipped.append(f"{label}: {column}")
            continue
        series[label] = metric_series
    missing = [label for label, _column in series_specs if label not in series]
    if missing:
        raise SystemExit(f"{generator} plot is missing required series: {missing}. Details: {skipped}")
    _assert_same_x_values(series, f"{generator}_baseline_vs_milp.pdf")
    _save_line_plot(
        output_dir / f"{generator}_baseline_vs_milp.pdf",
        "Synthetic validity stress test: baseline vs shard-aware MILP",
        "sequence length L",
        "Mean certified budget B*",
        series,
        stagger_coincident_markers=False,
    )


def plot_validity_demo_csv(csv_path: str, save_dir: str | None = None) -> list[dict[str, object]]:
    """Read a validity_demo benchmark CSV and write only validity_demo PDF plots."""
    path = Path(csv_path)
    output_dir = Path(save_dir) if save_dir is not None else path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows_csv(path)
    _validate_validity_demo_plot_rows(rows, csv_path=path)
    save_validity_demo_plot(rows, output_dir, generator="validity_demo")
    demo_rows = [row for row in rows if row.get("generator") == "validity_demo"]
    write_validity_demo_parameter_tables(output_dir, demo_rows, csv_path=path)
    curve_checks = save_validity_demo_budget_curve_plots(output_dir, source_dir=path.parent, result_rows=demo_rows)
    write_validity_demo_audit(
        output_dir / "audit_validity_demo.md",
        demo_rows,
        csv_path=path,
        generator="validity_demo",
        curve_checks=curve_checks,
        budget_rows=_read_optional_csv(path.parent / "benchmark_budget_curves.csv"),
    )
    print(f"Wrote validity_demo plots under: {output_dir}")
    return rows


def write_validity_demo_parameter_tables(
    output_dir: Path,
    rows: list[dict[str, object]],
    *,
    csv_path: Path,
) -> None:
    """Record the exact K/N/L/T combinations observed in validity-demo results."""
    grouped: dict[tuple[int, int, int, int], list[dict[str, object]]] = {}
    for row in rows:
        values = [_numeric_value(row.get(key)) for key in ["K", "N", "L", "T"]]
        if any(value is None for value in values):
            continue
        key = tuple(int(value) for value in values)
        grouped.setdefault(key, []).append(row)

    table_rows: list[dict[str, object]] = []
    for (K, N, L, T), matching in sorted(grouped.items()):
        q1_statuses = sorted(
            {str(row["row_col_val_q1_status"]) for row in matching if row.get("row_col_val_q1_status") not in {None, ""}}
        )
        qn_statuses = sorted(
            {str(row["row_col_val_qN_status"]) for row in matching if row.get("row_col_val_qN_status") not in {None, ""}}
        )
        table_rows.append(
            {
                "K": K,
                "N": N,
                "L": L,
                "T": T,
                "result_rows": len(matching),
                "q1_status": ",".join(q1_statuses),
                "qN_status": ",".join(qn_statuses),
            }
        )

    csv_output = output_dir / "validity_demo_parameters.csv"
    _write_rows_csv(csv_output, table_rows)
    markdown_lines = [
        "# Validity demo parameters used",
        "",
        f"Source results: `{csv_path}`",
        "",
        f"Observed result rows: {len(rows)}",
        f"Observed unique `(K, N, L, T)` combinations: {len(table_rows)}",
        "",
        f"- `K`: {_sorted_unique_values(rows, 'K')}",
        f"- `N`: {_sorted_unique_values(rows, 'N')}",
        f"- `L`: {_sorted_unique_values(rows, 'L')}",
        f"- `T`: {_sorted_unique_values(rows, 'T')}",
        "",
        "| K | N | L | T | Result rows | q1 status | qN status |",
        "|---:|---:|---:|---:|---:|:---|:---|",
    ]
    markdown_lines.extend(
        f"| {row['K']} | {row['N']} | {row['L']} | {row['T']} | {row['result_rows']} | "
        f"{row['q1_status'] or '-'} | {row['qN_status'] or '-'} |"
        for row in table_rows
    )
    (output_dir / "validity_demo_parameters.md").write_text("\n".join(markdown_lines) + "\n")


def _validate_validity_demo_plot_rows(rows: list[dict[str, object]], csv_path: Path) -> None:
    demo_rows = [row for row in rows if row.get("generator") == "validity_demo"]
    if not demo_rows:
        return
    l_values = _sorted_unique_values(demo_rows, "L")
    missing_l = []
    infeasible_l = []
    nonoptimal_l = []
    missing_methods: list[str] = []
    required_metrics = {
        "Plain DPA max-token phrase blocker": "plain_dpa_val_sequence_q1",
        "TPA max-token phrase baseline": "tpa_val_sequence_q1",
        "Shared shard-aware MILP full sequence": "row_col_val_q1",
    }
    for L in l_values:
        matching = [row for row in demo_rows if row.get("L") == L]
        for label, metric in required_metrics.items():
            if not any(_numeric_value(row.get(metric)) is not None for row in matching):
                missing_methods.append(f"{label} at L={L}")
        values = [_numeric_value(row.get("row_col_val_q1")) for row in matching]
        statuses = {row.get("row_col_val_q1_status") for row in matching}
        if not any(value is not None for value in values):
            missing_l.append(L)
            if "INFEASIBLE" in statuses:
                infeasible_l.append(L)
        if statuses and statuses != {"OPTIMAL"}:
            nonoptimal_l.append((L, sorted(str(status) for status in statuses)))
    if missing_l or nonoptimal_l or missing_methods:
        raise SystemExit(
            f"validity_demo shared MILP full-sequence results are missing for L={missing_l} in {csv_path}. "
            f"INFEASIBLE L values: {infeasible_l or 'none reported'}. "
            f"Non-OPTIMAL L statuses: {nonoptimal_l or 'none'}. "
            f"Missing plotted methods: {missing_methods or 'none'}. "
            "Regenerate data after fixing the validity_demo config/generator."
        )


def _assert_same_x_values(series: dict[str, tuple[list[float], list[float]]], plot_name: str) -> None:
    expected: list[float] | None = None
    for label, (xs, _ys) in series.items():
        if expected is None:
            expected = list(xs)
            continue
        if list(xs) != expected:
            raise SystemExit(f"{plot_name} would silently drop or misalign x-values for {label}: {xs} != {expected}")


def save_validity_demo_budget_curve_plots(
    output_dir: Path,
    source_dir: Path | None = None,
    result_rows: list[dict[str, object]] | None = None,
) -> dict[str, bool]:
    csv_dir = source_dir if source_dir is not None else output_dir
    budget_rows = _read_optional_csv(csv_dir / "benchmark_budget_curves.csv")
    budget_plot_num_points = _validity_demo_budget_plot_num_points(result_rows or [])
    return _save_required_certified_fraction_budget_plot(
        budget_rows,
        output_dir / "validity_demo_certified_fraction_by_budget.pdf",
        "validity_demo certified fraction by budget",
        [
            ("Shared shard-aware MILP full sequence", "Shared MILP", "validity_full_harmful_sequence_per_prompt", "radius_derived"),
            ("TPA max-token phrase baseline", "TPA max-token phrase baseline", "validity_full_harmful_sequence_per_prompt", "radius_derived"),
            ("Plain DPA max-token phrase blocker", "Plain DPA max-token phrase blocker", "validity_full_harmful_sequence_per_prompt", "radius_derived"),
        ],
        max_points=budget_plot_num_points,
        stagger_coincident_markers=False,
    )


def write_validity_demo_audit(
    path: Path,
    rows: list[dict[str, object]],
    csv_path: Path,
    generator: str,
    curve_checks: dict[str, bool],
    budget_rows: list[dict[str, object]] | None = None,
) -> None:
    gap_series = _mean_series_by_axis(rows, "L", "validity_gap_joint_minus_tpa_q1")
    if gap_series is None:
        gap_series = _computed_gap_series(rows, "row_col_val_q1", "tpa_val_sequence_q1")
    tpa_minus_dpa_series = _mean_series_by_axis(rows, "L", "validity_gap_tpa_minus_plain_dpa_q1")
    if tpa_minus_dpa_series is None:
        tpa_minus_dpa_series = _computed_gap_series(rows, "tpa_val_sequence_q1", "plain_dpa_val_sequence_q1")
    gap_observed = any((_numeric_value(row.get("validity_gap_joint_minus_tpa_q1")) or 0) > 0 for row in rows)
    if not gap_observed:
        gap_observed = any((_safe_numeric_difference(row.get("row_col_val_q1"), row.get("tpa_val_sequence_q1")) or 0) > 0 for row in rows)
    gap_grows = False
    if gap_series is not None and len(gap_series[1]) >= 2:
        gap_grows = gap_series[1][-1] > gap_series[1][0]
    ordering_observed = all(
        (_numeric_value(row.get("plain_dpa_val_sequence_q1")) or float("inf"))
        < (_numeric_value(row.get("tpa_val_sequence_q1")) or -float("inf"))
        < (_numeric_value(row.get("row_col_val_q1")) or -float("inf"))
        for row in rows
    )
    no_milp_drops = _validity_demo_no_milp_drops(rows)
    plotted_budget_points = curve_checks.get("plotted_budget_points")
    lines = [
        f"# {generator} synthetic validity stress-test audit",
        "",
        f"CSV used: {csv_path}",
        f"Generator: {generator}",
        "Scope: artificial controlled validity-only demo, not a natural language benchmark.",
        f"K values: {_sorted_unique_values(rows, 'K')}",
        f"N values: {_sorted_unique_values(rows, 'N')}",
        f"L values: {_sorted_unique_values(rows, 'L')}",
        f"T values: {_sorted_unique_values(rows, 'T')}",
        "",
        "## Main plotted budgets",
        f"Certified-fraction plotted budget points: {plotted_budget_points if plotted_budget_points is not None else 'all available'}",
        "",
        "Plain DPA max-token phrase blocker by L:",
        *_format_mean_series(rows, "plain_dpa_val_sequence_q1"),
        "",
        "TPA max-token phrase baseline values by L:",
        *_format_mean_series(rows, "tpa_val_sequence_q1"),
        "",
        "Shared shard-aware MILP full sequence values by L:",
        *_format_mean_series(rows, "row_col_val_q1"),
        "",
        "## Token diagnostics",
        "Per-token vote count examples by L, row 0:",
        *_format_first_value_by_axis(rows, "validity_demo_vote_count_examples_row0"),
        "",
        "Plain DPA token radii by L, row 0:",
        *_format_first_value_by_axis(rows, "validity_demo_plain_dpa_token_radii_row0"),
        "",
        "TPA token radii by L, row 0:",
        *_format_first_value_by_axis(rows, "validity_demo_tpa_token_radii_row0"),
        "",
        "Plain DPA token radii summary by L:",
        *_format_first_value_by_axis(rows, "validity_demo_plain_token_radii_summary"),
        "",
        "TPA token radii summary by L:",
        *_format_first_value_by_axis(rows, "validity_demo_tpa_token_radii_summary"),
        "",
        "Plain DPA phrase radii summary by L:",
        *_format_first_value_by_axis(rows, "validity_demo_plain_phrase_radii_summary"),
        "",
        "TPA phrase radii summary by L:",
        *_format_first_value_by_axis(rows, "validity_demo_tpa_phrase_radii_summary"),
        "",
        "Shared MILP per-row phrase radii summary by L:",
        *_format_budget_radius_summary_by_l(budget_rows or [], "Shared MILP", "validity_full_harmful_sequence_per_prompt"),
        "",
        "Cell percentages where TPA compares to Plain DPA by L:",
        *_format_tpa_plain_percentages(rows),
        "",
        "Phrase-level TPA max-token radii by L, row 0:",
        *_format_first_value_by_axis(rows, "validity_demo_tpa_phrase_radius_row0"),
        "",
        "Shard group assignments by token position, row 0:",
        *_format_first_value_by_axis(rows, "validity_demo_shard_groups_row0"),
        "",
        "Union size of required shard groups by L:",
        *_format_first_value_by_axis(rows, "validity_demo_required_group_union_size"),
        "",
        "## MILP diagnostics",
        "Shared MILP q1 status by L:",
        *_format_status_series(rows, "row_col_val_q1_status"),
        "",
        "Shared MILP q1 value by L:",
        *_format_mean_series(rows, "row_col_val_q1"),
        "",
        "Joint minus TPA gap by L:",
        *(_format_series(gap_series) if gap_series is not None else ["- no numeric rows"]),
        "",
        "Shared MILP relative lift over TPA by L:",
        *_format_relative_lift_by_axis(rows, "row_col_val_q1", "tpa_val_sequence_q1"),
        "",
        "Shared MILP mean-radius minus TPA mean-radius by L:",
        *_format_budget_mean_gap_by_l(budget_rows or [], "Shared MILP", "TPA max-token phrase baseline", "validity_full_harmful_sequence_per_prompt"),
        "",
        "Shared MILP mean-radius relative lift over TPA by L:",
        *_format_budget_mean_relative_lift_by_l(
            budget_rows or [],
            "Shared MILP",
            "TPA max-token phrase baseline",
            "validity_full_harmful_sequence_per_prompt",
        ),
        "",
        "TPA minus plain DPA count-margin gap by L:",
        *(_format_series(tpa_minus_dpa_series) if tpa_minus_dpa_series is not None else ["- no numeric rows"]),
        "",
        "TPA relative lift over plain DPA by L:",
        *_format_relative_lift_by_axis(rows, "tpa_val_sequence_q1", "plain_dpa_val_sequence_q1"),
        "",
        "## Fail-fast confirmations",
        f"Budget curves monotone non-increasing: {curve_checks.get('monotone_nonincreasing', False)}",
        f"No MILP values silently dropped: {no_milp_drops}",
        f"Shared MILP q1 status OPTIMAL for every plotted L: {_all_status_optimal(rows, 'row_col_val_q1_status')}",
        f"Expected Plain DPA max-token phrase blocker < TPA max-token phrase baseline < Shared shard-aware MILP ordering observed: {ordering_observed}",
        f"Expected gap observed: {gap_observed}",
        f"Gap grows with L: {gap_grows}",
        "",
        "Generated cells individually feasible under intended shard group by L:",
        *_format_first_value_by_axis(rows, "validity_demo_group_feasible_all"),
        "",
        "## Explanation",
        f"{generator} is artificial and controlled. It is not intended to model a natural language distribution.",
        "Plain DPA max-token phrase blocker only reads the top-vs-target count margin at each token, so it misses the cost of overtaking many tied competitors.",
        "TPA is count-based and sees each harmful target token as individually cheap after targeted count transfer.",
        "The full shared MILP is shard-aware and must use one shared poisoned-shard allocation across target positions.",
        "The demo assigns cheap target-token attacks to different shard groups, so the full harmful sequence requires more shared poisoned shards than TPA's count-only sequence baseline suggests.",
    ]
    path.write_text("\n".join(lines))
    _write_validity_demo_comparison_report(path.parent / "comparisons.txt", rows, csv_path, budget_rows or [])


def _format_status_series(rows: list[dict[str, object]], metric: str, axis_name: str = "L") -> list[str]:
    grouped: dict[float, set[str]] = {}
    for row in rows:
        axis_value = _numeric_value(row.get(axis_name))
        status = row.get(metric)
        if axis_value is None or status in {None, ""}:
            continue
        grouped.setdefault(axis_value, set()).add(str(status))
    if not grouped:
        return ["- no rows"]
    return [f"- {axis_name}={x:g}: {', '.join(sorted(grouped[x]))}" for x in sorted(grouped)]


def _all_status_optimal(rows: list[dict[str, object]], metric: str) -> bool:
    statuses = [row.get(metric) for row in rows]
    return bool(statuses) and all(status == "OPTIMAL" for status in statuses)


def _validity_demo_no_milp_drops(rows: list[dict[str, object]]) -> bool:
    l_values = _sorted_unique_values(rows, "L")
    for L in l_values:
        matching = [row for row in rows if row.get("L") == L]
        if not matching:
            return False
        if not all(row.get("row_col_val_q1_status") == "OPTIMAL" for row in matching):
            return False
        if not all(_numeric_value(row.get("row_col_val_q1")) is not None for row in matching):
            return False
    return True


def _format_first_value_by_axis(rows: list[dict[str, object]], metric: str, axis_name: str = "L") -> list[str]:
    grouped: dict[float, object] = {}
    for row in rows:
        axis_value = _numeric_value(row.get(axis_name))
        value = row.get(metric)
        if axis_value is None or value in {None, ""} or axis_value in grouped:
            continue
        grouped[axis_value] = value
    if not grouped:
        return ["- no rows"]
    return [f"- {axis_name}={x:g}: {grouped[x]}" for x in sorted(grouped)]


def _format_tpa_plain_percentages(rows: list[dict[str, object]]) -> list[str]:
    grouped: dict[float, dict[str, float]] = {}
    for row in rows:
        axis_value = _numeric_value(row.get("L"))
        if axis_value is None:
            continue
        gt = _numeric_value(row.get("validity_demo_pct_tpa_gt_plain"))
        eq = _numeric_value(row.get("validity_demo_pct_tpa_eq_plain"))
        lt = _numeric_value(row.get("validity_demo_pct_tpa_lt_plain"))
        if gt is None or eq is None or lt is None:
            continue
        grouped[axis_value] = {"gt": gt, "eq": eq, "lt": lt}
    if not grouped:
        return ["- no rows"]
    return [
        f"- L={x:g}: TPA>Plain {grouped[x]['gt']:.3g}%, TPA=Plain {grouped[x]['eq']:.3g}%, TPA<Plain {grouped[x]['lt']:.3g}%"
        for x in sorted(grouped)
    ]


def _write_validity_demo_comparison_report(path: Path, rows: list[dict[str, object]], csv_path: Path, budget_rows: list[dict[str, object]]) -> None:
    lines = [
        "Validity demo strategy comparisons",
        "",
        f"CSV file used: {csv_path}",
        "Generated by plotting only: yes",
        "Reruns Gurobi or regenerates experiments: no",
        "",
        "Math:",
        "absolute lift = max(0, mean(strategy A) - mean(strategy B))",
        "relative lift = 100 * absolute_lift / abs(mean(strategy B))",
        f"relative lifts below {RELATIVE_LIFT_PERCENT_REPORTING_THRESHOLD:g}% are reported as 0%",
        "",
        "Mean certified-budget comparisons by L:",
        "Shared MILP over TPA:",
        *_format_relative_lift_by_axis(rows, "row_col_val_q1", "tpa_val_sequence_q1"),
        "",
        "TPA over Plain DPA:",
        *_format_relative_lift_by_axis(rows, "tpa_val_sequence_q1", "plain_dpa_val_sequence_q1"),
        "",
        "Mean per-row radius comparisons by L:",
        "Shared MILP over TPA:",
        *_format_budget_mean_relative_lift_by_l(
            budget_rows,
            "Shared MILP",
            "TPA max-token phrase baseline",
            "validity_full_harmful_sequence_per_prompt",
        ),
        "",
        "Token-level TPA vs Plain DPA percentages:",
        *_format_tpa_plain_percentages(rows),
        "",
        "Raw mean budgets by L:",
        "Plain DPA:",
        *_format_mean_series(rows, "plain_dpa_val_sequence_q1"),
        "",
        "TPA:",
        *_format_mean_series(rows, "tpa_val_sequence_q1"),
        "",
        "Shared MILP:",
        *_format_mean_series(rows, "row_col_val_q1"),
    ]
    path.write_text("\n".join(lines))


def _format_budget_radius_summary_by_l(rows: list[dict[str, object]], method: str, objective: str) -> list[str]:
    selected = [
        row
        for row in rows
        if row.get("method") == method
        and row.get("objective") == objective
        and row.get("curve_type") == "radius_derived"
        and _numeric_value(row.get("budget")) == 0
    ]
    if not selected:
        return ["- no rows"]
    lines = []
    for row in sorted(selected, key=lambda item: _numeric_value(item.get("L")) or 0):
        l_value = _numeric_value(row.get("L"))
        if l_value is None:
            continue
        lines.append(
            f"- L={l_value:g}: min={_numeric_value(row.get('min_radius')):.6g},"
            f"median={_numeric_value(row.get('median_radius')):.6g},"
            f"mean={_numeric_value(row.get('mean_radius')):.6g},"
            f"max={_numeric_value(row.get('max_radius')):.6g}"
        )
    return lines or ["- no rows"]


def _format_budget_mean_gap_by_l(rows: list[dict[str, object]], left_method: str, right_method: str, objective: str) -> list[str]:
    means: dict[tuple[str, float], float] = {}
    for row in rows:
        if row.get("objective") != objective or row.get("curve_type") != "radius_derived":
            continue
        if _numeric_value(row.get("budget")) != 0:
            continue
        method = str(row.get("method"))
        if method not in {left_method, right_method}:
            continue
        l_value = _numeric_value(row.get("L"))
        mean = _numeric_value(row.get("mean_radius"))
        if l_value is None or mean is None:
            continue
        means[(method, l_value)] = mean
    l_values = sorted({l_value for method, l_value in means if method in {left_method, right_method}})
    lines = []
    for l_value in l_values:
        left = means.get((left_method, l_value))
        right = means.get((right_method, l_value))
        if left is not None and right is not None:
            lines.append(f"- L={l_value:g}: {left - right:.6g}")
    return lines or ["- no rows"]


def _format_budget_mean_relative_lift_by_l(rows: list[dict[str, object]], left_method: str, right_method: str, objective: str) -> list[str]:
    means: dict[tuple[str, float], float] = {}
    for row in rows:
        if row.get("objective") != objective or row.get("curve_type") != "radius_derived":
            continue
        if _numeric_value(row.get("budget")) != 0:
            continue
        method = str(row.get("method"))
        if method not in {left_method, right_method}:
            continue
        l_value = _numeric_value(row.get("L"))
        mean = _numeric_value(row.get("mean_radius"))
        if l_value is None or mean is None:
            continue
        means[(method, l_value)] = mean
    l_values = sorted({l_value for method, l_value in means if method in {left_method, right_method}})
    lines = []
    for l_value in l_values:
        left = means.get((left_method, l_value))
        right = means.get((right_method, l_value))
        if left is None or right is None:
            continue
        lift, percent = _relative_lift(left, right)
        lines.append(f"- L={l_value:g}: +{lift:.6g} mean budget units ({percent:.6g}%)")
    return lines or ["- no rows"]


def _format_relative_lift_by_axis(rows: list[dict[str, object]], left_metric: str, right_metric: str, axis_name: str = "L") -> list[str]:
    grouped: dict[float, list[tuple[float, float]]] = {}
    for row in rows:
        axis_value = _numeric_value(row.get(axis_name))
        left = _numeric_value(row.get(left_metric))
        right = _numeric_value(row.get(right_metric))
        if axis_value is None or left is None or right is None:
            continue
        grouped.setdefault(axis_value, []).append((left, right))
    lines = []
    for axis_value in sorted(grouped):
        left_mean = float(np.mean([left for left, _right in grouped[axis_value]]))
        right_mean = float(np.mean([right for _left, right in grouped[axis_value]]))
        lift, percent = _relative_lift(left_mean, right_mean)
        lines.append(f"- {axis_name}={axis_value:g}: +{lift:.6g} mean budget units ({percent:.6g}%)")
    return lines or ["- no rows"]


def _relative_lift(left: float, right: float) -> tuple[float, float]:
    lift = max(0.0, left - right)
    return lift, _relative_lift_percent(lift, right)


def _relative_lift_percent(lift: float, baseline: float) -> float:
    if lift <= 1e-12 or abs(baseline) <= 1e-12:
        return 0.0
    percent = float(100.0 * lift / abs(baseline))
    if percent < RELATIVE_LIFT_PERCENT_REPORTING_THRESHOLD:
        return 0.0
    return percent


def _computed_gap_series(rows: list[dict[str, object]], left_metric: str, right_metric: str) -> tuple[list[float], list[float]] | None:
    grouped: dict[float, list[float]] = {}
    for row in rows:
        x_value = _numeric_value(row.get("L"))
        diff = _safe_numeric_difference(row.get(left_metric), row.get(right_metric))
        if x_value is None or diff is None:
            continue
        grouped.setdefault(x_value, []).append(diff)
    if not grouped:
        return None
    xs, ys = [], []
    for x in sorted(grouped):
        xs.append(x)
        ys.append(float(np.mean(grouped[x])))
    return xs, ys


def _safe_numeric_difference(left: object, right: object) -> float | None:
    left_value = _numeric_value(left)
    right_value = _numeric_value(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _format_series(series: tuple[list[float], list[float]]) -> list[str]:
    xs, ys = series
    return [f"- L={x:g}: {y:.6g}" for x, y in zip(xs, ys)]

def _print_instance_summary(
    data: ToyData, K: int, N: int, L: int, T: int, delta_stab: float, delta_val: float, target_bias: float, seed: int, influence_mode: str
) -> None:
    margins = stability_margins(data.stab_counts, data.clean_pred)
    print(
        f"K={K}, N={N}, L={L}, T={T}, delta_stab={delta_stab}, delta_val={delta_val}, "
        f"target_bias={target_bias}, influence_mode={influence_mode}, seed={seed}"
    )
    print()
    print("clean predictions:")
    print(data.clean_pred)
    print()
    print("harmful targets:")
    print(data.target)
    print()
    print("winner-vs-runner-up margins:")
    print(margins)


def _resolve_objective_flags(
    *,
    objective_family: str,
    make_budget_curves: bool,
    make_stability_objectives: bool | None,
    make_validity_objectives: bool | None,
    make_stability_budget_curves: bool | None,
    make_validity_budget_curves: bool | None,
) -> dict[str, bool]:
    if objective_family == "full":
        flags = {
            "make_stability_objectives": True,
            "make_validity_objectives": True,
            "make_stability_budget_curves": make_budget_curves,
            "make_validity_budget_curves": make_budget_curves,
        }
    elif objective_family == "validity_only":
        flags = {
            "make_stability_objectives": False,
            "make_validity_objectives": True,
            "make_stability_budget_curves": False,
            "make_validity_budget_curves": make_budget_curves,
        }
    else:
        raise ValueError(f"Unknown objective_family: {objective_family}")
    for key, value in {
        "make_stability_objectives": make_stability_objectives,
        "make_validity_objectives": make_validity_objectives,
        "make_stability_budget_curves": make_stability_budget_curves,
        "make_validity_budget_curves": make_validity_budget_curves,
    }.items():
        if value is not None:
            flags[key] = value
    return flags


def _compute_benchmark_baselines(data: ToyData, make_stability_objectives: bool, make_validity_objectives: bool) -> dict[str, int | float]:
    if make_stability_objectives and make_validity_objectives:
        return compute_reference_baselines(data)
    rows: dict[str, int | float] = {}
    if make_validity_objectives:
        validity_cell_budgets = _cell_validity_budgets(data).astype(float)
        validity_cell_budgets[validity_cell_budgets > data.val_votes.shape[0]] = np.nan
        plain_dpa_cell_budgets = _plain_dpa_validity_token_budgets(data)
        targeted_validity_cell_budgets = _targeted_validity_token_budgets(data)
        phrase_row_budgets = _atomic_phrase_validity_row_budgets(data)
        row_validity_weak_radii = _row_nanmin_or_unknown(validity_cell_budgets)
        row_validity_strong_radii = _row_nanmax_all_known(validity_cell_budgets)
        independent_validity_row_costs = np.sum(validity_cell_budgets, axis=1)
        rows.update(
            {
                "dpa_val_cell_min": _finite_int_min(validity_cell_budgets),
                "dpa_val_row_weak_q1": _finite_int_min(row_validity_weak_radii),
                "dpa_val_row_weak_qN": _finite_int_max(row_validity_weak_radii),
                "dpa_val_row_strong_q1": _finite_int_min(row_validity_strong_radii),
                "dpa_val_row_strong_qN": _finite_int_max(row_validity_strong_radii),
                "raw_dpa_val_min_cell": _finite_int_min(validity_cell_budgets),
                "plain_dpa_val_cell_min": int(np.min(plain_dpa_cell_budgets)),
                **aggregate_plain_dpa_sequence_baselines(plain_dpa_cell_budgets),
                "tpa_val_cell_min": int(np.min(targeted_validity_cell_budgets)),
                **aggregate_tpa_sequence_baselines(targeted_validity_cell_budgets),
                "independent_val_sequence_q1": _finite_int_min(independent_validity_row_costs),
                "independent_val_sequence_qN": _finite_int_sum(independent_validity_row_costs),
                "independent_val_q1": _finite_int_min(independent_validity_row_costs),
                "independent_val_qN": _finite_int_sum(independent_validity_row_costs),
                "phrase_dpa_val_q1": int(np.min(phrase_row_budgets)),
                "phrase_dpa_val_qN": int(np.max(phrase_row_budgets)),
                "phrase_independent_val_q1": int(np.min(phrase_row_budgets)),
                "phrase_independent_val_qN": int(phrase_row_budgets.sum()),
            }
        )
    if make_stability_objectives:
        stability_cell_budgets = _cell_stability_budgets(data)
        row_stability_radii = stability_cell_budgets.min(axis=1)
        independent_stability_row_costs = stability_cell_budgets.sum(axis=1)
        rows.update(
            {
                "raw_dpa_stab_min_cell": int(np.min(_phd_margin_stability_budgets(data))),
                "dpa_stab_cell_min": int(np.min(stability_cell_budgets)),
                "dpa_stab_row_radius_q1": int(np.min(row_stability_radii)),
                "dpa_stab_row_radius_qN": int(np.max(row_stability_radii)),
                "independent_stab_full_row_q1": int(np.min(independent_stability_row_costs)),
                "independent_stab_full_row_qN": int(independent_stability_row_costs.sum()),
                "independent_stab_qN_rL": int(independent_stability_row_costs.sum()),
            }
        )
    return rows


def _solve_benchmark_certificates(
    data: ToyData,
    T: int,
    make_stability_objectives: bool = True,
    make_validity_objectives: bool = True,
    gurobi_threads: int | None = None,
) -> list[CertificateResult]:
    """Solve the certificate set stored in benchmark CSV rows."""
    N = data.stab_votes.shape[1]
    L = data.stab_votes.shape[2]
    results: list[CertificateResult] = []
    if make_stability_objectives:
        results.extend(
            [
                solve_structured_stability(
                    data.stab_votes,
                    data.stab_counts,
                    data.clean_pred,
                    data.influence,
                    q_rows=1,
                    r_cols=1,
                    gurobi_threads=gurobi_threads,
                ),
                solve_structured_stability(
                    data.stab_votes,
                    data.stab_counts,
                    data.clean_pred,
                    data.influence,
                    q_rows=1,
                    r_cols=L,
                    gurobi_threads=gurobi_threads,
                ),
                solve_structured_stability(
                    data.stab_votes,
                    data.stab_counts,
                    data.clean_pred,
                    data.influence,
                    q_rows=N,
                    r_cols=1,
                    gurobi_threads=gurobi_threads,
                ),
                solve_structured_stability(
                    data.stab_votes,
                    data.stab_counts,
                    data.clean_pred,
                    data.influence,
                    q_rows=N,
                    r_cols=L,
                    gurobi_threads=gurobi_threads,
                ),
            ]
        )
    if make_validity_objectives:
        results.extend(
            [
                solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=1, gurobi_threads=gurobi_threads),
                solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=N, gurobi_threads=gurobi_threads),
            ]
        )
    return results


def certified_fraction_from_radii(radii: np.ndarray, budgets: Iterable[int]) -> list[dict[str, int | float]]:
    """Compute strict ``B < B*`` certification summaries for each budget.

    Non-finite radii are unknown. They remain in ``num_total`` but are not
    counted as certified; summary radius statistics are computed over known
    finite radii only.
    """
    radii = np.asarray(radii, dtype=float)
    if radii.ndim != 1:
        raise ValueError("radii must be one-dimensional")
    if radii.size == 0:
        raise ValueError("radii must be non-empty")
    known = np.isfinite(radii)
    known_radii = radii[known]
    num_total = int(radii.size)
    num_known = int(np.sum(known))
    num_unknown = num_total - num_known
    rows = []
    for budget in budgets:
        certified = known & (budget < radii)
        num_certified = int(np.sum(certified))
        certified_fraction = num_certified / num_total
        rows.append(
            {
                "budget": int(budget),
                "certified_fraction": float(certified_fraction),
                "attacked_fraction": float(1.0 - certified_fraction),
                "mean_radius": float(np.mean(known_radii)) if num_known else float("nan"),
                "median_radius": float(np.median(known_radii)) if num_known else float("nan"),
                "min_radius": float(np.min(known_radii)) if num_known else float("nan"),
                "max_radius": float(np.max(known_radii)) if num_known else float("nan"),
                "num_certified": num_certified,
                "num_known": num_known,
                "num_unknown": num_unknown,
                "num_total": num_total,
            }
        )
    return rows


def _row_nanmin_or_unknown(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[0], np.nan, dtype=float)
    for idx, row in enumerate(values):
        finite = row[np.isfinite(row)]
        if finite.size:
            result[idx] = float(np.min(finite))
    return result


def _row_nanmax_all_known(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[0], np.nan, dtype=float)
    for idx, row in enumerate(values):
        if np.all(np.isfinite(row)):
            result[idx] = float(np.max(row))
    return result


def _finite_int_min(values: np.ndarray) -> int | float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return int(np.min(finite)) if finite.size else float("nan")


def _finite_int_max(values: np.ndarray) -> int | float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return int(np.max(finite)) if finite.size else float("nan")


def _finite_int_sum(values: np.ndarray) -> int | float:
    finite = np.asarray(values, dtype=float)
    return int(np.sum(finite)) if np.all(np.isfinite(finite)) else float("nan")


def compute_radius_derived_budget_curve_rows(
    data: ToyData,
    T: int,
    budgets: Iterable[int],
    metadata: dict[str, object],
    make_stability_curves: bool = True,
    make_validity_curves: bool = True,
    gurobi_threads: int | None = None,
) -> list[dict[str, object]]:
    """Build long-format radius-derived certified-fraction curve rows."""
    del T
    curves = []
    if make_stability_curves:
        stability_cell_budgets = _cell_stability_budgets(data)
        curves.extend(
            [
                (
                    "DPA token margin",
                    "full_response_stable_against_any_token_change",
                    stability_cell_budgets.min(axis=1),
                ),
                (
                    "Shared MILP",
                    "stability_one_token_per_prompt",
                    _shared_stability_row_radii(data, r_cols=1, gurobi_threads=gurobi_threads),
                ),
                (
                    "Shared MILP",
                    "stability_full_sequence_per_prompt",
                    _shared_stability_row_radii(
                        data,
                        r_cols=data.stab_votes.shape[2],
                        gurobi_threads=gurobi_threads,
                    ),
                ),
            ]
        )
    if make_validity_curves:
        validity_cell_budgets = _cell_validity_budgets(data).astype(float)
        validity_cell_budgets[validity_cell_budgets > data.val_votes.shape[0]] = np.nan
        plain_dpa_cell_budgets = _plain_dpa_validity_token_budgets(data)
        targeted_validity_cell_budgets = _targeted_validity_token_budgets(data)
        phrase_row_budgets = _atomic_phrase_validity_row_budgets(data)
        curves.extend(
            [
                (
                    "Plain DPA max-token phrase blocker",
                    "validity_full_harmful_sequence_per_prompt",
                    plain_dpa_cell_budgets.max(axis=1),
                ),
                (
                    "Shard-aware weakest-token diagnostic",
                    "weakest_harmful_token_not_full_sequence_validity",
                    _row_nanmin_or_unknown(validity_cell_budgets),
                ),
                (
                    "Shard-aware independent max-token diagnostic",
                    "most_difficult_harmful_token_not_full_sequence_validity",
                    _row_nanmax_all_known(validity_cell_budgets),
                ),
                (
                    "TPA max-token phrase baseline",
                    "validity_full_harmful_sequence_per_prompt",
                    targeted_validity_cell_budgets.max(axis=1),
                ),
                (
                    "Shared MILP",
                    "validity_full_harmful_sequence_per_prompt",
                    _shared_validity_row_radii(data, gurobi_threads=gurobi_threads),
                ),
                (
                    "Atomic phrase aggregation",
                    "validity_full_harmful_sequence_per_prompt",
                    phrase_row_budgets,
                ),
                (
                    "Independent shard-aware composition diagnostic",
                    "validity_full_harmful_sequence_per_prompt",
                    validity_cell_budgets.sum(axis=1),
                ),
            ]
        )
    rows: list[dict[str, object]] = []
    for method, objective, radii in curves:
        for summary in certified_fraction_from_radii(radii, budgets):
            rows.append(
                {
                    **metadata,
                    **summary,
                    "method": method,
                    "objective": objective,
                    "curve_type": "radius_derived",
                }
            )
    return rows


def compute_structured_stability_grid(data: ToyData, gurobi_threads: int | None = None) -> np.ndarray:
    """Compute ``B*(q,r)`` for every prompt-row/token-position stability objective."""
    N = data.stab_votes.shape[1]
    L = data.stab_votes.shape[2]
    grid = np.zeros((N, L), dtype=np.int64)
    for q_rows in range(1, N + 1):
        for r_cols in range(1, L + 1):
            result = solve_structured_stability(
                data.stab_votes,
                data.stab_counts,
                data.clean_pred,
                data.influence,
                q_rows=q_rows,
                r_cols=r_cols,
                gurobi_threads=gurobi_threads,
            )
            grid[q_rows - 1, r_cols - 1] = -1 if result.B_star is None else result.B_star
    return grid


def compute_validity_q_curve(data: ToyData, T: int, gurobi_threads: int | None = None) -> list[int]:
    """Compute validity budgets for one harmful sequence through all prompts."""
    N = data.val_votes.shape[1]
    values = []
    for q_rows in range(1, N + 1):
        result = solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=q_rows, gurobi_threads=gurobi_threads)
        values.append(-1 if result.B_star is None else result.B_star)
    return values


def _shared_stability_row_radii(data: ToyData, r_cols: int, gurobi_threads: int | None = None) -> np.ndarray:
    _, N, _ = data.stab_votes.shape
    radii = np.full(N, np.nan, dtype=float)
    for i in range(N):
        row_slice = slice(i, i + 1)
        result = solve_structured_stability(
            data.stab_votes[:, row_slice, :],
            data.stab_counts[row_slice, :, :],
            data.clean_pred[row_slice, :],
            data.influence[:, row_slice, :],
            q_rows=1,
            r_cols=r_cols,
            gurobi_threads=gurobi_threads,
        )
        if result.is_optimal and result.B_star is not None:
            radii[i] = float(result.B_star)
    return radii


def _shared_validity_row_radii(data: ToyData, gurobi_threads: int | None = None) -> np.ndarray:
    _, N, _ = data.val_votes.shape
    T = data.val_counts.shape[2]
    radii = np.full(N, np.nan, dtype=float)
    for i in range(N):
        row_slice = slice(i, i + 1)
        result = solve_row_col_validity(
            data.val_votes[:, row_slice, :],
            data.val_counts[row_slice, :, :],
            data.target[row_slice, :],
            T,
            data.influence[:, row_slice, :],
            q_rows=1,
            gurobi_threads=gurobi_threads,
        )
        if result.is_optimal and result.B_star is not None:
            radii[i] = float(result.B_star)
    return radii


def _benchmark_metadata(
    seed: int,
    K: int,
    N: int,
    L: int,
    T: int,
    delta_stab: float,
    delta_val: float,
    target_bias: float,
    influence_mode: str,
) -> dict[str, object]:
    return {
        "seed": seed,
        "K": K,
        "N": N,
        "L": L,
        "T": T,
        "delta_stab": delta_stab,
        "delta_val": delta_val,
        "target_bias": target_bias,
        "influence_mode": influence_mode,
    }


def _save_required_certified_fraction_budget_plot(
    rows: list[dict[str, object]],
    path: Path,
    title: str,
    selections: list[tuple[str, str, str, str]],
    max_points: int | None = None,
    stagger_coincident_markers: bool = True,
) -> dict[str, bool]:
    if not rows:
        raise SystemExit(f"{path.name} requires benchmark_budget_curves.csv rows, but none were available.")
    series, skipped = _budget_curve_series(rows, selections)
    missing = [label for label, _method, _objective, _curve_type in selections if label not in series]
    if missing:
        raise SystemExit(f"{path.name} is missing required certified-fraction series: {missing}. Details: {skipped}")
    _assert_same_x_values(series, path.name)
    monotone = all(_is_nonincreasing(ys) for _xs, ys in series.values())
    if not monotone:
        raise SystemExit(f"{path.name} has a non-monotone certified-fraction curve.")
    if max_points is not None:
        series = _sample_budget_plot_series(series, max_points=max_points)
        _assert_same_x_values(series, path.name)
    _save_line_plot(
        path,
        title,
        "Poisoned shard budget B",
        "Certified fraction (%)",
        series,
        stagger_coincident_markers=stagger_coincident_markers,
    )
    point_counts = {len(xs) for xs, _ys in series.values()}
    plotted_points = point_counts.pop() if len(point_counts) == 1 else None
    return {"monotone_nonincreasing": monotone, "missing_required_series": False, "plotted_budget_points": plotted_points}


def _validity_demo_budget_plot_num_points(rows: list[dict[str, object]]) -> int | None:
    values = {
        int(value)
        for row in rows
        if (value := _numeric_value(row.get("budget_plot_num_points"))) is not None
    }
    if not values:
        return None
    if len(values) != 1:
        raise SystemExit(f"validity_demo has inconsistent budget_plot_num_points values: {sorted(values)}")
    value = values.pop()
    if value < 2:
        raise SystemExit("validity_demo budget_plot_num_points must be at least 2")
    return value


def _sample_budget_plot_series(
    series: dict[str, tuple[list[float], list[float]]],
    *,
    max_points: int,
) -> dict[str, tuple[list[float], list[float]]]:
    if not series:
        return series
    first_xs = next(iter(series.values()))[0]
    if len(first_xs) <= max_points:
        return series
    selected_indexes = np.linspace(0, len(first_xs) - 1, num=max_points, dtype=int)
    indexes = sorted(set(int(index) for index in selected_indexes))
    if indexes[0] != 0:
        indexes.insert(0, 0)
    if indexes[-1] != len(first_xs) - 1:
        indexes.append(len(first_xs) - 1)
    sampled: dict[str, tuple[list[float], list[float]]] = {}
    for label, (xs, ys) in series.items():
        sampled[label] = ([xs[index] for index in indexes], [ys[index] for index in indexes])
    return sampled


def _mean_series_by_axis(rows: list[dict[str, object]], axis_name: str, metric: str) -> tuple[list[float], list[float]] | None:
    """Group numeric row values by an x-axis column and return mean y-values."""
    grouped: dict[float, list[float]] = {}
    for row in rows:
        x_value = _numeric_value(row.get(axis_name))
        y_value = _numeric_value(row.get(metric))
        if x_value is None or y_value is None:
            continue
        grouped.setdefault(x_value, []).append(y_value)
    if not grouped:
        return None
    xs, ys = [], []
    for x in sorted(grouped):
        xs.append(x)
        ys.append(float(np.mean(grouped[x])))
    return xs, ys


def _axis_label(axis_name: str) -> str:
    if axis_name == "L":
        return "sequence length L"
    return axis_name


def _numeric_value(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _save_heatmap_pdf(
    matrix: np.ndarray,
    path: Path,
    title: str,
    fmt: str = ".0f",
    x_label: str = "token column j",
    y_label: str = "prompt row i",
    colorbar_label: str | None = None,
) -> None:
    """Save a report-ready PDF heatmap."""
    _prepare_pdf_path(path)
    rows, cols = matrix.shape
    values = matrix.astype(float)
    cmap = LinearSegmentedColormap.from_list("certificate", ["#2563eb", "#14b8a6", "#eab308"])
    fig_width = max(5.0, min(16.0, 1.0 + 0.65 * cols))
    fig_height = max(4.0, min(14.0, 1.8 + 0.58 * rows))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(values, cmap=cmap, aspect="auto")
    ax.set_title(title, fontsize=11, wrap=True, pad=14)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(np.arange(cols), labels=[j + 1 if colorbar_label else j for j in range(cols)])
    ax.set_yticks(np.arange(rows), labels=[i + 1 if colorbar_label else i for i in range(rows)])
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    midpoint = (float(np.min(values)) + float(np.max(values))) / 2
    for i in range(rows):
        for j in range(cols):
            text_color = "white" if float(values[i, j]) <= midpoint else "#111827"
            ax.text(j, i, format(float(values[i, j]), fmt), ha="center", va="center", color=text_color, fontsize=9)
    if colorbar_label is not None:
        colorbar = fig.colorbar(image, ax=ax, pad=0.03)
        colorbar.set_label(colorbar_label)
    fig.savefig(path, format="pdf", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_line_plot(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: dict[str, tuple[list[float], list[float]]],
    *,
    stagger_coincident_markers: bool = True,
    ranges: dict[str, tuple[list[float], list[float], list[float]]] | None = None,
) -> None:
    """Save a report-facing line plot as PDF."""
    _save_line_plot_pdf(
        path,
        title,
        x_label,
        y_label,
        series,
        stagger_coincident_markers=stagger_coincident_markers,
        ranges=ranges,
    )


def _save_line_plot_pdf(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: dict[str, tuple[list[float], list[float]]],
    *,
    stagger_coincident_markers: bool = True,
    ranges: dict[str, tuple[list[float], list[float], list[float]]] | None = None,
) -> None:
    """Save a multi-series PDF line plot with visible coincident series."""
    _prepare_pdf_path(path)
    all_x = [x for xs, _ in series.values() for x in xs]
    all_y = [y for _, ys in series.values() for y in ys]
    if not all_x or not all_y:
        return
    xmin, xmax = min(all_x), max(all_x)
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2", "#be123c", "#4d7c0f", "#9333ea", "#475569"]
    markers = ["o", "s", "D", "^"]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    point_groups: dict[tuple[float, float], list[int]] = {}
    for series_idx, (_name, (xs, ys)) in enumerate(series.items()):
        for x, y in zip(xs, ys):
            point_groups.setdefault((float(x), float(y)), []).append(series_idx)

    for idx, (name, (xs, ys)) in enumerate(series.items()):
        color = CANONICAL_COLORS.get(name, colors[idx % len(colors)])
        line_style = _line_style_for_index(name, idx)
        if ranges and name in ranges:
            range_xs, lower, upper = ranges[name]
            ax.fill_between(range_xs, lower, upper, color=color, alpha=0.14, linewidth=0)
        ax.plot(xs, ys, color=color, linestyle=line_style, linewidth=2.2, label=name)
        x_span = xmax - xmin if xmax != xmin else 1.0
        for x, y in zip(xs, ys):
            point_key = (float(x), float(y))
            overlapping_indices = point_groups[point_key]
            overlap_position = overlapping_indices.index(idx)
            marker_x = _coincident_marker_x(
                x,
                overlap_position=overlap_position,
                overlap_count=len(overlapping_indices),
                x_span=x_span,
                stagger=stagger_coincident_markers,
            )
            ax.scatter(
                [marker_x],
                [y],
                marker=markers[idx % len(markers)],
                s=48,
                facecolors="white",
                edgecolors=color,
                linewidths=1.8,
                zorder=3 + idx,
            )

    ax.set_title(title, fontsize=13, wrap=True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(bottom=min(0.0, min(all_y)))
    ax.grid(axis="y", color="#e5e7eb")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    if stagger_coincident_markers and any(len(indices) > 1 for indices in point_groups.values()):
        fig.text(
            0.99,
            0.02,
            "Coincident markers are offset for visibility.",
            ha="right",
            fontsize=8,
            color="#4b5563",
        )
    fig.savefig(path, format="pdf", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _coincident_marker_x(
    x: float,
    *,
    overlap_position: int,
    overlap_count: int,
    x_span: float,
    stagger: bool,
) -> float:
    if not stagger:
        return x
    return x + (overlap_position - (overlap_count - 1) / 2) * 0.012 * x_span


def _prepare_pdf_path(path: Path) -> None:
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"plot output must use a .pdf extension: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _line_style_for_index(label: str, series_index: int = 0) -> str | tuple[int, tuple[int, ...]]:
    if label == "Plain DPA max-token phrase blocker":
        return ":"
    if label in {"DPA most difficult harmful token", "Shard-aware independent max-token diagnostic"}:
        return ":"
    if "baseline" in label:
        return "--"
    dash_patterns = [
        "-",
        "--",
        ":",
        "-.",
    ]
    return dash_patterns[series_index % len(dash_patterns)]


def _parse_scalar_config_value(value: str) -> object:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        return [_parse_scalar_config_value(part) for part in value[1:-1].split(",") if part.strip()]
    unquoted = value.strip("'\"")
    lower = unquoted.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        return int(unquoted)
    except ValueError:
        try:
            return float(unquoted)
        except ValueError:
            return unquoted


def _read_flat_yaml_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"config file not found: {config_path}")
    config: dict[str, object] = {}
    current_section: str | None = None
    for line_number, raw_line in enumerate(config_path.read_text().splitlines(), start=1):
        uncommented = raw_line.split("#", maxsplit=1)[0]
        line = uncommented.strip()
        if not line:
            continue
        if ":" not in line:
            raise ConfigError(f"invalid config line {line_number} in {config_path}: expected `key: value`")
        key, value = line.split(":", maxsplit=1)
        key = key.strip()
        if not key:
            raise ConfigError(f"invalid config line {line_number} in {config_path}: empty key")
        is_indented = uncommented[: len(uncommented) - len(uncommented.lstrip())] != ""
        if is_indented:
            if current_section is None:
                raise ConfigError(f"invalid config line {line_number} in {config_path}: nested field without section")
            key = f"{current_section}.{key}"
        else:
            current_section = None
        if key == "preset":
            raise ConfigError(f"field `preset` in {config_path} is not supported in YAML configs; use `name` for metadata")
        if value.strip() == "":
            if is_indented:
                raise ConfigError(f"invalid config line {line_number} in {config_path}: nested sections are not supported")
            current_section = key
            continue
        config[key] = _parse_scalar_config_value(value)
    return config


def _require_field(config: dict[str, object], path: Path, key: str) -> object:
    if key not in config:
        raise ConfigError(f"missing required field `{key}` in {path}")
    return config[key]


def _require_str(config: dict[str, object], path: Path, key: str, allowed: set[str] | None = None) -> str:
    value = _require_field(config, path, key)
    if not isinstance(value, str):
        raise ConfigError(f"field `{key}` in {path} must be a string")
    if allowed is not None and value not in allowed:
        raise ConfigError(f"field `{key}` in {path} must be one of {sorted(allowed)}")
    return value


def _require_int(config: dict[str, object], path: Path, key: str) -> int:
    value = _require_field(config, path, key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"field `{key}` in {path} must be an integer")
    return value


def _require_bool(config: dict[str, object], path: Path, key: str) -> bool:
    value = _require_field(config, path, key)
    if not isinstance(value, bool):
        raise ConfigError(f"field `{key}` in {path} must be a boolean")
    return value


def _optional_str(config: dict[str, object], path: Path, key: str, default: str, allowed: set[str] | None = None) -> str:
    if key not in config:
        return default
    value = config[key]
    if not isinstance(value, str):
        raise ConfigError(f"field `{key}` in {path} must be a string")
    if allowed is not None and value not in allowed:
        raise ConfigError(f"field `{key}` in {path} must be one of {sorted(allowed)}")
    return value


def _optional_int(config: dict[str, object], path: Path, key: str, default: int) -> int:
    if key not in config:
        return default
    value = config[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"field `{key}` in {path} must be an integer")
    return int(value)


def _optional_nonnegative_int(config: dict[str, object], path: Path, key: str, default: int) -> int:
    value = _optional_int(config, path, key, default)
    if value < 0:
        raise ConfigError(f"field `{key}` in {path} must be >= 0; use 0 for Gurobi automatic mode")
    return value


def _resolve_config_gurobi_threads(config: dict[str, object], path: Path) -> int:
    yaml_threads = _optional_nonnegative_int(config, path, "solver.gurobi_threads", 0)
    env_value = os.environ.get("GUROBI_THREADS")
    if env_value is None:
        return yaml_threads
    # Benchmark config priority is GUROBI_THREADS env var > YAML
    # solver.gurobi_threads > default 0. Direct solver calls can still pass an
    # explicit gurobi_threads argument, which has priority inside milp.py.
    try:
        return resolve_gurobi_threads(None)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _optional_bool(config: dict[str, object], path: Path, key: str, default: bool) -> bool:
    if key not in config:
        return default
    value = config[key]
    if not isinstance(value, bool):
        raise ConfigError(f"field `{key}` in {path} must be a boolean")
    return bool(value)


def _require_number_list(config: dict[str, object], path: Path, key: str, item_type: type) -> list[int] | list[float]:
    value = _require_field(config, path, key)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"field `{key}` in {path} must be a non-empty list")
    if item_type is int:
        if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            raise ConfigError(f"field `{key}` in {path} must be a list of integers")
        return [int(item) for item in value]
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        raise ConfigError(f"field `{key}` in {path} must be a list of numbers")
    return [float(item) for item in value]


def load_experiment_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path)
    config = _read_flat_yaml_config(config_path)
    if "stability_competitor_mode" in config:
        raise ConfigError(
            f"deprecated field `stability_competitor_mode` found in {config_path}. "
            "Stability now always uses all competitors. Remove this field from the YAML."
        )
    generator = _require_str(config, config_path, "generator", {"toy", "validity_demo"})
    if "seed_values" in config and "seed" in config:
        raise ConfigError(f"{config_path} must define either `seed_values` or `seed`, not both")
    seeds = (
        _require_number_list(config, config_path, "seed_values", int)
        if "seed_values" in config
        else [_require_int(config, config_path, "seed")]
    )
    benchmark_config: dict[str, object] = {
        "Ks": _require_number_list(config, config_path, "K_values", int),
        "Ns": _require_number_list(config, config_path, "N_values", int),
        "lengths": _require_number_list(config, config_path, "L_values", int),
        "Ts": _require_number_list(config, config_path, "T_values", int),
        "generator": generator,
        "seeds": seeds,
        "budget_max": _require_int(config, config_path, "budget_max"),
        "save_dir": _require_str(config, config_path, "output_dir"),
        "objective_family": _require_str(config, config_path, "objective_family", {"full", "validity_only"}),
        "make_budget_curves": _require_bool(config, config_path, "make_budget_curves"),
        "gurobi_threads": _resolve_config_gurobi_threads(config, config_path),
    }
    if generator == "toy":
        benchmark_config.update(
            {
                "delta_stabs": _require_number_list(config, config_path, "delta_stab_values", float),
                "delta_vals": _require_number_list(config, config_path, "delta_val_values", float),
                "target_biases": _require_number_list(config, config_path, "target_bias_values", float),
                "influence_mode": _require_str(config, config_path, "influence_mode", {"dense", "row-local", "column-local"}),
            }
        )
    else:
        benchmark_config.update(
            {
                "delta_stabs": [0.0],
                "delta_vals": [0.0],
                "target_biases": [0.0],
                "influence_mode": _optional_str(config, config_path, "influence_mode", "validity_demo", {"validity_demo"}),
            }
        )
    for key in [
        "make_stability_objectives",
        "make_validity_objectives",
        "make_stability_budget_curves",
        "make_validity_budget_curves",
    ]:
        if key in config:
            value = config[key]
            if not isinstance(value, bool):
                raise ConfigError(f"field `{key}` in {config_path} must be a boolean")
            benchmark_config[key] = value
    if generator == "validity_demo":
        group_size = _require_int(config, config_path, "group_size")
        overlap = _require_int(config, config_path, "overlap")
        stride = group_size - overlap
        if stride <= 0:
            raise ConfigError(f"field `overlap` in {config_path} must be smaller than group_size")
        benchmark_config.update(
            {
                "group_size": group_size,
                "overlap": overlap,
                "validity_demo_distribution": _optional_str(
                    config,
                    config_path,
                    "validity_demo_distribution",
                    "deterministic",
                    {"deterministic", "heterogeneous"},
                ),
                "num_competitor_min": _optional_int(config, config_path, "num_competitor_min", 2),
                "num_competitor_max": _optional_int(config, config_path, "num_competitor_max", 8),
                "target_count_min": _optional_int(config, config_path, "target_count_min", 0),
                "target_count_max": _optional_int(config, config_path, "target_count_max", 3),
                "competitor_gap_min": _optional_int(config, config_path, "competitor_gap_min", 2),
                "competitor_gap_max": _optional_int(config, config_path, "competitor_gap_max", min(6, group_size)),
                "competitor_jitter": _optional_int(config, config_path, "competitor_jitter", 2),
                "row_difficulty_jitter": _optional_bool(config, config_path, "row_difficulty_jitter", True),
                "position_difficulty_jitter": _optional_bool(config, config_path, "position_difficulty_jitter", True),
                "budget_plot_num_points": _optional_int(config, config_path, "budget_plot_num_points", 0),
            }
        )
        if "target_gap" in config:
            benchmark_config["target_gap"] = _require_int(config, config_path, "target_gap")
        if benchmark_config["budget_plot_num_points"] == 0:
            benchmark_config["budget_plot_num_points"] = None
    return benchmark_config


def _validity_demo_min_required_shards(L: int, group_size: int, overlap: int) -> int:
    """Return the shard prefix needed by the first ``L`` validity-demo groups."""
    return group_size + (L - 1) * (group_size - overlap)


def _estimate_solve_counts(
    Ks: Iterable[int],
    Ns: Iterable[int],
    Ls: Iterable[int],
    Ts: Iterable[int],
    delta_stabs: Iterable[float],
    delta_vals: Iterable[float],
    target_biases: Iterable[float],
    seeds: Iterable[int],
    _budget_max: int,
    objective_flags: dict[str, bool],
) -> dict[str, int]:
    counts = {
        "instances": 0,
        "stability_objective_solves": 0,
        "validity_objective_solves": 0,
        "per_row_stability_solves": 0,
        "per_row_validity_solves": 0,
    }
    for _seed in seeds:
        for _K in Ks:
            for N in Ns:
                for _L in Ls:
                    for _T in Ts:
                        for _delta_stab in delta_stabs:
                            for _delta_val in delta_vals:
                                for _target_bias in target_biases:
                                    counts["instances"] += 1
                                    if objective_flags["make_stability_objectives"]:
                                        counts["stability_objective_solves"] += 4
                                    if objective_flags["make_validity_objectives"]:
                                        counts["validity_objective_solves"] += 2
                                    if objective_flags["make_stability_budget_curves"]:
                                        counts["per_row_stability_solves"] += 2 * int(N)
                                    if objective_flags["make_validity_budget_curves"]:
                                        counts["per_row_validity_solves"] += int(N)
    return counts


def _print_benchmark_dry_run(
    *,
    Ks: Iterable[int],
    Ns: Iterable[int],
    Ls: Iterable[int],
    Ts: Iterable[int],
    delta_stabs: Iterable[float],
    delta_vals: Iterable[float],
    target_biases: Iterable[float],
    generator: str,
    objective_family: str,
    objective_flags: dict[str, bool],
    budget_max: int,
    save_dir: str,
    verbose: bool,
    group_size: int,
    overlap: int,
    seeds: Iterable[int],
) -> None:
    Ks, Ns, Ls, Ts = list(Ks), list(Ns), list(Ls), list(Ts)
    seeds = list(seeds)
    delta_stabs, delta_vals, target_biases = list(delta_stabs), list(delta_vals), list(target_biases)
    counts = _estimate_solve_counts(Ks, Ns, Ls, Ts, delta_stabs, delta_vals, target_biases, seeds, budget_max, objective_flags)
    total_stability = counts["stability_objective_solves"] + counts["per_row_stability_solves"]
    total_validity = counts["validity_objective_solves"] + counts["per_row_validity_solves"]
    print("Benchmark dry run")
    print(f"generator: {generator}")
    print(f"output_dir: {save_dir}")
    print(f"objective_family: {objective_family}")
    print(f"stability objectives: {'enabled' if objective_flags['make_stability_objectives'] else 'disabled'}")
    print(f"validity objectives: {'enabled' if objective_flags['make_validity_objectives'] else 'disabled'}")
    print(f"stability budget curves: {'enabled' if objective_flags['make_stability_budget_curves'] else 'disabled'}")
    print(f"validity budget curves: {'enabled' if objective_flags['make_validity_budget_curves'] else 'disabled'}")
    print(f"expanded benchmark instances: {counts['instances']}")
    print(f"seed values: {seeds}")
    K_master_requested = max(Ks)
    N_master = max(Ns)
    L_master = max(Ls)
    T_master = max(Ts)
    K_master = (
        max(K_master_requested, _validity_demo_min_required_shards(L_master, group_size, overlap))
        if generator == "validity_demo"
        else K_master_requested
    )
    print("Coupled generation enabled.")
    print("Master dimensions for this config:")
    print(f"  K_master = {K_master}")
    print(f"  N_master = {N_master}")
    print(f"  L_master = {L_master}")
    print(f"  T_master = {T_master}")
    print(
        "Master random generations: "
        f"{len(seeds) * len(delta_stabs) * len(delta_vals) * len(target_biases)} "
        "(one per seed/delta_stab/delta_val/target_bias group)"
    )
    if verbose:
        print(f"K values: {Ks}")
        print(f"N values: {Ns}")
        print(f"L values: {Ls}")
        print(f"T values: {Ts}")
        print(f"delta_stab values: {delta_stabs}")
        print(f"delta_val values: {delta_vals}")
        print(f"target_bias values: {target_biases}")
        print("Derived instances:")
        for K in Ks:
            for N in Ns:
                for L in Ls:
                    for T in Ts:
                        if generator == "validity_demo":
                            minimum = _validity_demo_min_required_shards(L, group_size, overlap)
                            K_actual = max(K, minimum)
                            print(
                                f"  K_requested={K}, K_actual={K_actual}, N={N}, L={L}, T={T}, "
                                f"min_required_shards={minimum}"
                            )
                        else:
                            print(f"  K_requested={K}, K_actual={K}, N={N}, L={L}, T={T}")
    print("estimated Gurobi solves:")
    print(f"  estimated stability solves: {total_stability}")
    print(f"  estimated validity solves: {total_validity}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for toy experiment workflows."""
    parser = argparse.ArgumentParser(description="Toy row/column certificate experiments.")
    parser.add_argument(
        "command",
        choices=[
            "visualize",
            "benchmark",
            "plot-csv",
            "plot-validity-demo",
            "plot-sweep",
        ],
    )
    parser.add_argument("--K", type=int, default=7)
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument("--delta-stab", type=float, default=None)
    parser.add_argument("--delta-val", type=float, default=None)
    parser.add_argument("--target-bias", type=float, default=None)
    parser.add_argument("--influence-mode", choices=["dense", "row-local", "column-local", "validity_demo"], default="dense")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--csv", default="toy_experiments/outputs/medium/results/benchmark_results.csv")
    parser.add_argument("--sweep", choices=["K", "N", "L", "degenerate"], default=None)
    parser.add_argument("--make-plots", action="store_true", help="Also render benchmark plots after running Gurobi.")
    parser.add_argument("--config", default=None, help="YAML benchmark config. Required for benchmark runs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print solve estimates without running Gurobi.")
    parser.add_argument("--verbose", action="store_true", help="Print resolved dry-run grid values.")
    return parser


def main() -> None:
    """Entry point for ``python -m toy_experiments.experiments``."""
    args = build_parser().parse_args()
    delta_stab = args.delta if args.delta_stab is None else args.delta_stab
    delta_val = args.delta if args.delta_val is None else args.delta_val
    target_bias = args.target_bias if args.target_bias is not None else 0.2
    if args.command == "visualize":
        visualize_instance(
            K=args.K,
            N=args.N,
            L=args.L,
            T=args.T,
            delta_stab=delta_stab,
            delta_val=delta_val,
            target_bias=target_bias,
            seed=args.seed,
            influence_mode=args.influence_mode,
            save_dir=args.save_dir or "toy_experiments/outputs/smoke",
        )
    elif args.command == "benchmark":
        if args.config is None:
            raise SystemExit("ConfigError: benchmark requires --config path/to/config.yaml")
        try:
            config = load_experiment_config(args.config)
        except ConfigError as exc:
            raise SystemExit(f"ConfigError: {exc}") from exc
        benchmark_scale(
            Ks=config["Ks"],
            Ns=config["Ns"],
            Ls=config["lengths"],
            Ts=config["Ts"],
            delta_stabs=config["delta_stabs"],
            delta_vals=config["delta_vals"],
            target_biases=config["target_biases"],
            influence_mode=config["influence_mode"],
            seed=config["seeds"],
            save_dir=config["save_dir"],
            make_plots=args.make_plots,
            budget_max=config["budget_max"],
            make_budget_curves=config["make_budget_curves"],
            objective_family=config["objective_family"],
            make_stability_objectives=config.get("make_stability_objectives"),
            make_validity_objectives=config.get("make_validity_objectives"),
            make_stability_budget_curves=config.get("make_stability_budget_curves"),
            make_validity_budget_curves=config.get("make_validity_budget_curves"),
            dry_run=args.dry_run,
            verbose=args.verbose,
            generator=config["generator"],
            group_size=config.get("group_size", 3),
            target_gap=config.get("target_gap", 1),
            overlap=config.get("overlap", 0),
            validity_demo_distribution=config.get("validity_demo_distribution", "deterministic"),
            num_competitor_min=config.get("num_competitor_min", 2),
            num_competitor_max=config.get("num_competitor_max", 8),
            target_count_min=config.get("target_count_min", 0),
            target_count_max=config.get("target_count_max", 3),
            competitor_gap_min=config.get("competitor_gap_min", 2),
            competitor_gap_max=config.get("competitor_gap_max", 6),
            competitor_jitter=config.get("competitor_jitter", 2),
            row_difficulty_jitter=config.get("row_difficulty_jitter", True),
            position_difficulty_jitter=config.get("position_difficulty_jitter", True),
            budget_plot_num_points=config.get("budget_plot_num_points"),
            gurobi_threads=config["gurobi_threads"],
        )
    elif args.command == "plot-csv":
        plot_benchmark_csv(args.csv, save_dir=args.save_dir)
    elif args.command == "plot-validity-demo":
        plot_validity_demo_csv(args.csv, save_dir=args.save_dir)
    elif args.command == "plot-sweep":
        if args.sweep is None:
            raise SystemExit("plot-sweep requires --sweep K|N|L|degenerate")
        plot_sweep_csv(args.csv, sweep=args.sweep, save_dir=args.save_dir)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
