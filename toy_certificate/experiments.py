"""Experiment CLI, benchmark generation, baselines, and SVG plotting.

This module is the orchestration layer for the first-party toy certificate
implementation. It builds synthetic :class:`toy_certificate.data.ToyData`
instances, calls the shared MILP solvers, writes benchmark CSVs, computes
DPA/TPA/atomic-phrase/independent-composition baselines, and renders PNG/SVG
plots. The external ``phd_reference/`` tree is not imported or modified here.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter

import numpy as np

from .baselines import (
    aggregate_tpa_sequence_baselines,
    atomic_phrase_validity_row_budgets as _atomic_phrase_validity_row_budgets,
    cell_stability_budgets as _cell_stability_budgets,
    cell_validity_budgets as _cell_validity_budgets,
    compute_reference_baselines,
    min_budget_from_contributions as _min_budget_from_contributions,
    min_budget_satisfying_all as _min_budget_satisfying_all,
    phd_margin_stability_budgets as _phd_margin_stability_budgets,
    targeted_partition_radius,
    targeted_validity_token_budgets as _targeted_validity_token_budgets,
)
from .csv_io import (
    copy_legacy_csv_columns as _copy_legacy_csv_columns,
    looks_numeric as _looks_numeric,
    read_optional_csv as _read_optional_csv,
    read_rows_csv as _read_rows_csv,
    write_rows_csv as _write_rows_csv,
)
from .data import ToyData, generate_toy_votes, generate_validity_demo_votes, stability_margins
from .milp import (
    CertificateResult,
    DamageResult,
    maximize_attacked_rows_stability,
    maximize_attacked_rows_validity,
    solve_col_stability,
    solve_col_validity,
    solve_row_col_stability,
    solve_row_col_validity,
    solve_row_stability,
    solve_row_validity,
    solve_structured_stability,
)


class ConfigError(ValueError):
    """Raised when a benchmark YAML config is missing or malformed."""


def run_sanity(
    K: int = 7,
    N: int = 3,
    L: int = 4,
    T: int = 5,
    delta_stab: float = 0.2,
    delta_val: float = 0.2,
    target_bias: float = 0.2,
    seed: int = 0,
    influence_mode: str = "dense",
    stability_competitor_mode: str = "all",
    show_grid: bool = False,
    save_dir: str | None = None,
) -> list[CertificateResult]:
    """Run one toy instance and print the default certificate table."""
    data = generate_toy_votes(
        K=K, N=N, L=L, T=T, delta_stab=delta_stab, delta_val=delta_val, target_bias=target_bias, seed=seed, influence_mode=influence_mode
    )
    _print_instance_summary(data, K, N, L, T, delta_stab, delta_val, target_bias, seed, influence_mode)
    if show_grid:
        print_console_grid(data)
    if save_dir is not None:
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
            stability_competitor_mode=stability_competitor_mode,
        )
    results = solve_default_certificates(data, T, stability_competitor_mode=stability_competitor_mode)
    print_certificate_table(results)
    return results


def solve_default_certificates(data: ToyData, T: int, stability_competitor_mode: str = "all") -> list[CertificateResult]:
    """Solve the small default certificate set used by the sanity command."""
    q_all_rows = data.stab_votes.shape[1]
    return [
        solve_row_stability(data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, competitor_mode=stability_competitor_mode),
        solve_col_stability(data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, competitor_mode=stability_competitor_mode),
        solve_row_col_stability(
            data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, definition="any_cell", competitor_mode=stability_competitor_mode
        ),
        solve_row_col_stability(
            data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, definition="full_row", competitor_mode=stability_competitor_mode
        ),
        solve_row_validity(data.val_votes, data.val_counts, data.target, T, data.influence),
        solve_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, definition="full_column"),
        solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=1),
        solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=q_all_rows),
    ]


def sweep_delta(K: int, N: int, L: int, T: int, deltas: Iterable[float], seed: int, stability_competitor_mode: str = "all") -> None:
    """Print a small delta sweep table for quick interactive debugging."""
    rows = []
    for delta in deltas:
        data = generate_toy_votes(K=K, N=N, L=L, T=T, delta_stab=delta, delta_val=delta, seed=seed)
        rows.append(
            [
                delta,
                solve_row_stability(data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, competitor_mode=stability_competitor_mode).B_star,
                solve_row_validity(data.val_votes, data.val_counts, data.target, T, data.influence).B_star,
                solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=1).B_star,
                solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=N).B_star,
            ]
        )
    _print_sweep_table(["delta", "row_stab", "row_val", "row_col_val_q1", "row_col_val_qN"], rows)


def sweep_length(K: int, N: int, lengths: Iterable[int], T: int, delta: float, seed: int, stability_competitor_mode: str = "all") -> None:
    """Print a small sequence-length sweep table."""
    rows = []
    for L in lengths:
        data = generate_toy_votes(K=K, N=N, L=L, T=T, delta_stab=delta, delta_val=delta, seed=seed)
        rows.append(
            [
                L,
                solve_row_stability(data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, competitor_mode=stability_competitor_mode).B_star,
                solve_row_validity(data.val_votes, data.val_counts, data.target, T, data.influence).B_star,
                solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=1).B_star,
                solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=N).B_star,
            ]
        )
    _print_sweep_table(["L", "row_stab", "row_val", "row_col_val_q1", "row_col_val_qN"], rows)


def sweep_prompts(K: int, prompts: Iterable[int], L: int, T: int, delta: float, seed: int, stability_competitor_mode: str = "all") -> None:
    """Print a small prompt-count sweep table."""
    rows = []
    for N in prompts:
        data = generate_toy_votes(K=K, N=N, L=L, T=T, delta_stab=delta, delta_val=delta, seed=seed)
        rows.append(
            [
                N,
                solve_row_stability(data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, competitor_mode=stability_competitor_mode).B_star,
                solve_row_validity(data.val_votes, data.val_counts, data.target, T, data.influence).B_star,
                solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=1).B_star,
                solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=N).B_star,
            ]
        )
    _print_sweep_table(["N", "row_stab", "row_val", "row_col_val_q1", "row_col_val_qN"], rows)


def visualize_instance(
    K: int,
    N: int,
    L: int,
    T: int,
    delta_stab: float,
    delta_val: float,
    target_bias: float,
    seed: int,
    influence_mode: str,
    save_dir: str,
    stability_competitor_mode: str = "all",
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
        stability_competitor_mode=stability_competitor_mode,
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
    stability_competitor_mode: str,
    seed: int,
    save_dir: str,
    make_plots: bool = False,
    budget_max: int = 15,
    make_budget_curves: bool = True,
    make_damage_curves: bool = True,
    make_horizon_curves: bool = True,
    objective_family: str = "full",
    make_stability_objectives: bool | None = None,
    make_validity_objectives: bool | None = None,
    make_stability_budget_curves: bool | None = None,
    make_validity_budget_curves: bool | None = None,
    make_stability_horizon_curves: bool | None = None,
    make_validity_horizon_curves: bool | None = None,
    dry_run: bool = False,
    verbose: bool = False,
    generator: str = "toy",
    group_size: int = 3,
    target_gap: int = 1,
    overlap: int = 0,
) -> list[dict[str, object]]:
    """Generate benchmark CSV rows for the configured parameter grid.

    CSV generation is intentionally separate from plotting. Stability solver
    rows record ``stability_competitor_mode`` so exact all-competitor runs can be
    distinguished from runner-up approximation runs.
    """
    objective_flags = _resolve_objective_flags(
        objective_family=objective_family,
        make_budget_curves=make_budget_curves,
        make_horizon_curves=make_horizon_curves,
        make_stability_objectives=make_stability_objectives,
        make_validity_objectives=make_validity_objectives,
        make_stability_budget_curves=make_stability_budget_curves,
        make_validity_budget_curves=make_validity_budget_curves,
        make_stability_horizon_curves=make_stability_horizon_curves,
        make_validity_horizon_curves=make_validity_horizon_curves,
    )
    objective_flags["make_damage_curves"] = make_damage_curves
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
        )
        return []
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    budget_curve_rows: list[dict[str, object]] = []
    damage_curve_rows: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []

    for K in Ks:
        for N in Ns:
            for L in Ls:
                for T in Ts:
                    for delta_stab in delta_stabs:
                        for delta_val in delta_vals:
                            for target_bias in target_biases:
                                if generator == "validity_demo":
                                    data = generate_validity_demo_votes(
                                        L=L,
                                        group_size=group_size,
                                        target_gap=target_gap,
                                        overlap=overlap,
                                        N=N,
                                        T=T,
                                        seed=seed,
                                        K=K,
                                    )
                                    K_actual = int(data.val_votes.shape[0])
                                elif generator == "toy":
                                    data = generate_toy_votes(
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
                                    K_actual = K
                                else:
                                    raise ValueError(f"Unknown generator: {generator}")
                                start = perf_counter()
                                results = _solve_benchmark_certificates(
                                    data,
                                    T,
                                    stability_competitor_mode=stability_competitor_mode,
                                    make_stability_objectives=objective_flags["make_stability_objectives"],
                                    make_validity_objectives=objective_flags["make_validity_objectives"],
                                )
                                runtime_total = perf_counter() - start
                                row = {
                                    "K": K_actual,
                                    "N": N,
                                    "L": L,
                                    "T": T,
                                    "generator": generator,
                                    "delta_stab": delta_stab,
                                    "delta_val": delta_val,
                                    "target_bias": target_bias,
                                    "seed": seed,
                                    "influence_mode": influence_mode,
                                    "stability_competitor_mode": stability_competitor_mode,
                                    "runtime_gurobi_total": f"{runtime_total:.6f}",
                                }
                                if generator == "validity_demo":
                                    row.update({"group_size": group_size, "target_gap": target_gap, "overlap": overlap})
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
                                _add_validity_demo_gap_columns(row)
                                rows.append(row)
                                budgets = list(range(0, min(K_actual, budget_max) + 1))
                                metadata = _benchmark_metadata(
                                    seed=seed,
                                    K=K_actual,
                                    N=N,
                                    L=L,
                                    T=T,
                                    delta_stab=delta_stab,
                                    delta_val=delta_val,
                                    target_bias=target_bias,
                                    influence_mode=influence_mode,
                                    stability_competitor_mode=stability_competitor_mode,
                                )
                                if objective_flags["make_stability_budget_curves"] or objective_flags["make_validity_budget_curves"]:
                                    budget_curve_rows.extend(
                                        compute_radius_derived_budget_curve_rows(
                                            data,
                                            T=T,
                                            budgets=budgets,
                                            metadata=metadata,
                                            stability_competitor_mode=stability_competitor_mode,
                                            make_stability_curves=objective_flags["make_stability_budget_curves"],
                                            make_validity_curves=objective_flags["make_validity_budget_curves"],
                                        )
                                    )
                                if make_damage_curves:
                                    damage_curve_rows.extend(
                                        compute_direct_damage_curve_rows(
                                            data,
                                            T=T,
                                            budgets=budgets,
                                            metadata=metadata,
                                            stability_competitor_mode=stability_competitor_mode,
                                            make_stability_curves=objective_flags["make_stability_objectives"],
                                            make_validity_curves=objective_flags["make_validity_objectives"],
                                        )
                                    )
                                if objective_flags["make_stability_horizon_curves"] or objective_flags["make_validity_horizon_curves"]:
                                    horizon_rows.extend(
                                        compute_horizon_curve_rows(
                                            data,
                                            budgets=budgets,
                                            metadata=metadata,
                                            make_stability_curves=objective_flags["make_stability_horizon_curves"],
                                            make_validity_curves=objective_flags["make_validity_horizon_curves"],
                                        )
                                    )
                                print(
                                    "bench "
                                    f"K={K_actual} N={N} L={L} T={T} delta_stab={delta_stab} delta_val={delta_val} target_bias={target_bias}: "
                                    + ", ".join(f"{result.name}={result.B_star}" for result in results)
                                )

    csv_path = output_dir / "benchmark_results.csv"
    _write_rows_csv(csv_path, rows)
    print()
    print(f"Wrote benchmark CSV: {csv_path}")
    if objective_flags["make_stability_budget_curves"] or objective_flags["make_validity_budget_curves"]:
        budget_csv_path = output_dir / "benchmark_budget_curves.csv"
        _write_rows_csv(budget_csv_path, budget_curve_rows)
        print(f"Wrote budget-curve CSV: {budget_csv_path}")
    if make_damage_curves:
        damage_csv_path = output_dir / "benchmark_damage_curves.csv"
        _write_rows_csv(damage_csv_path, damage_curve_rows)
        print(f"Wrote direct-damage CSV: {damage_csv_path}")
    if objective_flags["make_stability_horizon_curves"] or objective_flags["make_validity_horizon_curves"]:
        horizon_csv_path = output_dir / "benchmark_horizons.csv"
        _write_rows_csv(horizon_csv_path, horizon_rows)
        print(f"Wrote horizon CSV: {horizon_csv_path}")
    if make_plots:
        save_default_report_plots(rows, output_dir, csv_path=csv_path)
        print(f"Wrote benchmark plots under: {output_dir}")
    else:
        print("Skipped plots. Generate plots with: ./scripts/plot.sh")
    return rows


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
    row_milp = _numeric_value(row.get("row_validity"))
    tpa = _numeric_value(row.get("tpa_val_sequence_q1"))
    row["validity_gap_joint_minus_tpa_q1"] = _safe_difference(joint, tpa)
    row["validity_ratio_joint_over_tpa_q1"] = _safe_ratio(joint, tpa)
    row["validity_gap_row_minus_tpa_q1"] = _safe_difference(row_milp, tpa)
    row["validity_ratio_row_over_tpa_q1"] = _safe_ratio(row_milp, tpa)
    dpa_weak = _numeric_value(row.get("dpa_val_row_weak_q1"))
    row["validity_gap_tpa_minus_dpa_weak_q1"] = _safe_difference(tpa, dpa_weak)
    row["validity_gap_joint_minus_dpa_weak_q1"] = _safe_difference(joint, dpa_weak)


def plot_benchmark_csv(csv_path: str, save_dir: str | None = None) -> list[dict[str, object]]:
    """Read a benchmark CSV and regenerate the default report-facing plots."""
    path = Path(csv_path)
    output_dir = Path(save_dir) if save_dir is not None else path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows_csv(path)
    save_default_report_plots(rows, output_dir, csv_path=path)
    print(f"Wrote benchmark plots under: {output_dir}")
    return rows


DEFAULT_REPORT_PLOT_FILENAMES = (
    "main_stability_budget_curve.svg",
    "main_validity_budget_curve.svg",
    "stability_certificate_vs_K.svg",
    "validity_certificate_vs_K.svg",
)

DISABLED_DEFAULT_PLOTS = (
    "ablation_stability_row_column_joint",
    "ablation_validity_row_column_joint",
    "direct_damage_curve_joint_milp",
    "stability_diagnostic_references",
    "validity_diagnostic_references",
    "stability_full_matrix_main_comparison",
    "stability_one_sequence_main_comparison",
    "validity_all_prompts_main_comparison",
    "validity_one_sequence_main_comparison",
    "stability_horizon_by_budget",
    "stability_horizon_fraction_by_budget",
    "validity_horizon_by_budget",
    "validity_horizon_fraction_by_budget",
    "validity_sensitivity_target_bias",
)


def save_default_report_plots(rows: list[dict[str, object]], output_dir: Path, csv_path: Path) -> None:
    """Write only the simplified default report plot set."""
    _clean_default_plot_dir(output_dir)
    budget_rows = _read_optional_csv(csv_path.parent / "benchmark_budget_curves.csv")
    audit: list[dict[str, object]] = []

    budget_specs = [
        (
            "main_stability_budget_curve.svg",
            "Main stability budget curve",
            [
                ("Joint row-column MILP", "Shared MILP", "stability_full_sequence_per_prompt", "radius_derived"),
                ("DPA weakest-token stability baseline", "DPA token margin", "full_response_stable_against_any_token_change", "radius_derived"),
            ],
        ),
        (
            "main_validity_budget_curve.svg",
            "Main validity budget curve",
            [
                ("Joint row-column MILP", "Shared MILP", "validity_full_harmful_sequence_per_prompt", "radius_derived"),
                ("TPA sequence baseline", "TPA max-token sequence", "validity_full_harmful_sequence_per_prompt", "radius_derived"),
                ("DPA weakest harmful-token diagnostic", "DPA weakest harmful token", "weakest_harmful_token_not_full_sequence_validity", "radius_derived"),
            ],
        ),
    ]
    for filename, title, selections in budget_specs:
        series, skipped = _budget_curve_series(budget_rows, selections)
        if series:
            _save_line_plot(output_dir / filename, title, "Poisoned shard budget B", "Certified fraction (%)", series)
        else:
            print(f"Warning: skipped {filename}; no requested budget-curve series were available.")
        audit.append({"plot": filename, "series": list(series), "skipped": skipped})

    metric_specs = [
        (
            "stability_certificate_vs_K.svg",
            "Stability certificate vs K",
            MAIN_STABILITY_METRICS,
        ),
        (
            "validity_certificate_vs_K.svg",
            "Validity certificate vs K",
            MAIN_VALIDITY_METRICS,
        ),
    ]
    for filename, title, metrics in metric_specs:
        series, skipped = _metric_series(rows, "K", metrics)
        if series:
            _save_line_plot(output_dir / filename, title, _axis_label("K"), "Mean certified budget B*", series)
        else:
            print(f"Warning: skipped {filename}; no requested benchmark-result series were available.")
        audit.append({"plot": filename, "series": list(series), "skipped": skipped})

    _write_default_plot_audit(output_dir / "audit_plot_outputs.txt", csv_path, audit)


def _clean_default_plot_dir(output_dir: Path) -> None:
    """Remove old generated plot artifacts from the default plot directory."""
    for child in output_dir.iterdir():
        if child.is_file() and (child.suffix in {".png", ".svg"} or child.name.startswith("audit_")):
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
        selected = [
            row
            for row in rows
            if row.get("method") == method and row.get("objective") == objective and row.get("curve_type") == curve_type
        ]
        metric_series = _mean_series_by_axis(selected, "budget", "certified_fraction")
        if metric_series is None:
            skipped.append(f"{label}: missing method={method}, objective={objective}, curve_type={curve_type}")
            continue
        xs, ys = metric_series
        series[label] = (xs, [100.0 * y for y in ys])
    return series, skipped


def _metric_series(
    rows: list[dict[str, object]],
    axis_name: str,
    metrics: dict[str, str],
) -> tuple[dict[str, tuple[list[float], list[float]]], list[str]]:
    series: dict[str, tuple[list[float], list[float]]] = {}
    skipped: list[str] = []
    for label, metric in metrics.items():
        metric_series = _mean_series_by_axis(rows, axis_name, metric)
        if metric_series is None:
            skipped.append(f"{label}: missing or empty column {metric}")
            continue
        series[label] = metric_series
    return series, skipped


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
        skipped = item["skipped"]
        if skipped:
            lines.append(f"  skipped: {'; '.join(skipped)}")
    lines.extend(
        [
            "",
            "Plots intentionally disabled in the default workflow:",
            *[f"- {name}" for name in DISABLED_DEFAULT_PLOTS],
            "",
            "Removed paths:",
            "- Removed old ablation, diagnostic-reference, main-comparison, direct-damage plot, and horizon plot generator paths that were no longer used by plot.sh.",
            "- Kept shared SVG/line/heatmap helpers used by smoke visualization and validity_demo plotting.",
        ]
    )
    path.write_text("\n".join(lines))


AUDIT_CURVE_SELECTIONS: list[tuple[str, str, str, str]] = [
    ("DPA weakest token, radius-derived", "DPA token margin", "full_response_stable_against_any_token_change", "radius_derived"),
    ("Shared one-token-per-prompt, direct MILP", "Shared MILP", "stability_one_token_per_prompt", "direct_damage_milp"),
    ("Shared full-sequence-per-prompt, direct MILP", "Shared MILP", "stability_full_sequence_per_prompt", "direct_damage_milp"),
    ("TPA max-token sequence, radius-derived", "TPA max-token sequence", "validity_full_harmful_sequence_per_prompt", "radius_derived"),
    ("Shared full sequence, radius-derived", "Shared MILP", "validity_full_harmful_sequence_per_prompt", "radius_derived"),
    ("Shared full sequence, direct MILP", "Shared MILP", "validity_full_harmful_sequence_per_prompt", "direct_damage_milp"),
]


def audit_curve_csvs(csv_dir: str) -> list[dict[str, object]]:
    """Print diagnostics for budget, damage, and horizon sidecar CSVs."""
    base = Path(csv_dir)
    budget_rows = _read_optional_csv(base / "benchmark_budget_curves.csv")
    damage_rows = _read_optional_csv(base / "benchmark_damage_curves.csv")
    horizon_rows = _read_optional_csv(base / "benchmark_horizons.csv")
    combined = budget_rows + damage_rows
    diagnostics: list[dict[str, object]] = []

    print(f"Curve audit: {base}")
    print(f"Rows: budget={len(budget_rows)} damage={len(damage_rows)} horizon={len(horizon_rows)}")
    print()
    print("Plotted certified-fraction series")
    series_by_label: dict[str, dict[float, float]] = {}
    for label, method, objective, curve_type in AUDIT_CURVE_SELECTIONS:
        rows = _select_curve_rows(combined, method, objective, curve_type)
        source = "benchmark_damage_curves.csv" if curve_type == "direct_damage_milp" else "benchmark_budget_curves.csv"
        exact_rows = rows
        if curve_type == "direct_damage_milp":
            exact_rows = [row for row in rows if row.get("certified_fraction_is_exact") is True or row.get("is_optimal") is True]
        metric_series = _mean_series_by_axis(exact_rows, "budget", "certified_fraction")
        point_count = 0 if metric_series is None else len(metric_series[0])
        skipped = len(rows) - len(exact_rows)
        print(f"- {label}: source={source}, rows={len(rows)}, plotted_points={point_count}, skipped_nonoptimal={skipped}")
        if metric_series is not None:
            xs, ys = metric_series
            series_by_label[label] = {x: y for x, y in zip(xs, ys)}
        diagnostics.append(
            {
                "check": "series_source",
                "label": label,
                "source": source,
                "rows": len(rows),
                "plotted_points": point_count,
                "skipped_nonoptimal": skipped,
            }
        )

    print()
    print("Identical plotted series")
    for left_label, left_series in series_by_label.items():
        for right_label, right_series in series_by_label.items():
            if left_label >= right_label:
                continue
            shared_budgets = sorted(set(left_series) & set(right_series))
            if not shared_budgets:
                continue
            identical = all(np.isclose(left_series[budget], right_series[budget]) for budget in shared_budgets)
            if identical and len(left_series) == len(right_series) == len(shared_budgets):
                print(f"- identical: {left_label} == {right_label} over {len(shared_budgets)} budget point(s)")
                diagnostics.append({"check": "identical_series", "left_label": left_label, "right_label": right_label, "points": len(shared_budgets)})

    diagnostics.extend(_audit_direct_damage_rows(damage_rows))
    diagnostics.extend(_audit_horizon_rows(horizon_rows))
    if not diagnostics:
        print("No diagnostics produced.")
    return diagnostics


def compare_stability_modes(
    Ks: Iterable[int],
    Ns: Iterable[int],
    Ls: Iterable[int],
    Ts: Iterable[int],
    deltas: Iterable[float],
    target_bias: float,
    influence_mode: str,
    seed: int,
    save_dir: str,
) -> list[dict[str, object]]:
    """Compare exact all-competitor and runner-up stability modes.

    The diagnostic writes one row per structured stability objective and records
    ``B_star`` values, optimality flags, statuses, and runtimes. Negative optimal
    differences indicate a violation of the expected ``runner_up >= all``
    relationship and are printed as warnings.
    """
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for K in Ks:
        for N in Ns:
            for L in Ls:
                for T in Ts:
                    for delta in deltas:
                        data = generate_toy_votes(
                            K=K,
                            N=N,
                            L=L,
                            T=T,
                            delta_stab=delta,
                            delta_val=delta,
                            target_bias=target_bias,
                            seed=seed,
                            influence_mode=influence_mode,
                        )
                        objectives = [
                            ("one prompt, one token", 1, 1),
                            ("one prompt, full sequence", 1, L),
                            ("all prompts, one token each", N, 1),
                            ("all prompts, full matrix", N, L),
                        ]
                        for objective, q_rows, r_cols in objectives:
                            start = perf_counter()
                            all_result = solve_structured_stability(
                                data.stab_votes,
                                data.stab_counts,
                                data.clean_pred,
                                data.runner_up,
                                data.influence,
                                q_rows=q_rows,
                                r_cols=r_cols,
                                competitor_mode="all",
                            )
                            all_runtime = perf_counter() - start
                            start = perf_counter()
                            runner_result = solve_structured_stability(
                                data.stab_votes,
                                data.stab_counts,
                                data.clean_pred,
                                data.runner_up,
                                data.influence,
                                q_rows=q_rows,
                                r_cols=r_cols,
                                competitor_mode="runner_up",
                            )
                            runner_runtime = perf_counter() - start
                            diff = None
                            if all_result.B_star is not None and runner_result.B_star is not None:
                                diff = runner_result.B_star - all_result.B_star
                                if all_result.is_optimal and runner_result.is_optimal and diff < 0:
                                    print(f"Warning: runner_up < all for K={K} N={N} L={L} T={T} delta={delta} objective={objective}")
                            rows.append(
                                {
                                    "seed": seed,
                                    "K": K,
                                    "N": N,
                                    "L": L,
                                    "T": T,
                                    "delta_stab": delta,
                                    "delta_val": delta,
                                    "target_bias": target_bias,
                                    "influence_mode": influence_mode,
                                    "objective": objective,
                                    "B_star_all": all_result.B_star,
                                    "B_star_runner_up": runner_result.B_star,
                                    "diff": "" if diff is None else diff,
                                    "all_is_optimal": all_result.is_optimal,
                                    "runner_up_is_optimal": runner_result.is_optimal,
                                    "all_status_name": all_result.status_name,
                                    "runner_up_status_name": runner_result.status_name,
                                    "all_runtime_sec": f"{all_runtime:.6f}",
                                    "runner_up_runtime_sec": f"{runner_runtime:.6f}",
                                }
                            )

    csv_path = output_dir / "stability_mode_comparison.csv"
    _write_rows_csv(csv_path, rows)
    _save_stability_mode_comparison_plots(rows, output_dir)
    _print_stability_mode_comparison_summary(rows)
    print(f"Wrote stability mode comparison CSV: {csv_path}")
    return rows


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
    margins = stability_margins(data.stab_counts, data.clean_pred, data.runner_up)
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
    stability_competitor_mode: str = "all",
) -> None:
    """Write per-instance heatmaps and curves for one generated toy instance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    margins = stability_margins(data.stab_counts, data.clean_pred, data.runner_up)
    target_counts = np.take_along_axis(data.val_counts, data.target[:, :, None], axis=2)[:, :, 0]
    stability_grid = compute_structured_stability_grid(data, competitor_mode=stability_competitor_mode)
    q_curve = compute_validity_q_curve(data, T)

    title_suffix = (
        f"K={K}, N={N}, L={L}, T={T}, delta_stab={delta_stab}, delta_val={delta_val}, "
        f"target_bias={target_bias}, influence={influence_mode}, stability_competitor_mode={stability_competitor_mode}, seed={seed}"
    )
    _save_heatmap_svg(data.clean_pred, output_dir / "clean_predictions.svg", "Clean predictions | " + title_suffix)
    _save_heatmap_svg(data.target, output_dir / "harmful_targets.svg", "Harmful targets | " + title_suffix)
    _save_heatmap_svg(margins, output_dir / "stability_margins.svg", "Winner vs runner-up margins | " + title_suffix)
    _save_heatmap_svg(target_counts, output_dir / "validity_target_counts.svg", "Target validity vote counts | " + title_suffix)
    _save_heatmap_svg(
        stability_grid,
        output_dir / "structured_stability_heatmap.svg",
        "Structured stability poison budget | " + title_suffix,
        x_label="affected tokens per prompt",
        y_label="affected prompts",
        colorbar_label="poison budget B*",
    )
    _save_line_plot_svg(
        output_dir / "validity_q_curve.svg",
        "Row-column validity B*(q) | " + title_suffix,
        "q rows compromised",
        "Poison budget B*",
        {"shared MILP": ([float(q) for q in range(1, N + 1)], [float(v) for v in q_curve])},
    )

    print()
    print(f"Wrote instance plots under: {output_dir}")


CANONICAL_METHODS = (
    "Joint row-column MILP",
    "DPA weakest-token stability baseline",
    "TPA sequence baseline",
)

CANONICAL_COLORS = {
    "Joint row-column MILP": "#1f77b4",
    "DPA weakest-token stability baseline": "#d62728",
    "DPA weakest harmful-token diagnostic": "#8c564b",
    "TPA sequence baseline": "#2ca02c",
    "Row-only MILP ablation": "#9467bd",
    "Column-only MILP ablation": "#ff7f0e",
}

MAIN_STABILITY_METRICS = {
    "Joint row-column MILP": "row_col_stab_qN_rL",
    "DPA weakest-token stability baseline": "dpa_stab_row_radius_qN",
}

MAIN_VALIDITY_METRICS = {
    "Joint row-column MILP": "row_col_val_qN",
    "DPA weakest harmful-token diagnostic": "dpa_val_row_weak_qN",
    "TPA sequence baseline": "tpa_val_sequence_qN",
}

def save_validity_demo_plot(rows: list[dict[str, object]], output_dir: Path, csv_path: Path, generator: str) -> None:
    """Write the controlled validity demo plot when matching rows are present."""
    demo_rows = [row for row in rows if row.get("generator") == generator]
    if not demo_rows:
        return
    series_specs = [
        ("TPA sequence baseline", "tpa_val_sequence_q1"),
        ("Row-only shared MILP", "row_validity"),
        ("Joint row-column shared MILP", "row_col_val_q1"),
        ("DPA weakest harmful-token diagnostic", "dpa_val_row_weak_q1"),
    ]
    series: dict[str, tuple[list[float], list[float]]] = {}
    skipped: list[str] = []
    for label, column in series_specs:
        metric_series = _mean_series_by_axis(demo_rows, "L", column)
        if metric_series is None:
            skipped.append(f"{label}: {column}")
            print(f"Warning: skipping validity_demo series '{label}' because column '{column}' is missing or empty.")
            continue
        series[label] = metric_series
    plot_name = f"{generator}_baseline_vs_milp.svg"
    if len(series) >= 2:
        _save_line_plot(
            output_dir / plot_name,
            f"{generator} baseline vs shard-aware MILP",
            "sequence length L",
            "Mean certified budget B*",
            series,
        )
    else:
        print(f"Warning: skipped {plot_name}; fewer than two plottable series.")
    write_validity_demo_audit(output_dir / f"audit_{generator}.txt", demo_rows, csv_path=csv_path, plotted=list(series), skipped=skipped, generator=generator)


def plot_validity_demo_csv(csv_path: str, save_dir: str | None = None) -> list[dict[str, object]]:
    """Read a validity_demo benchmark CSV and write only validity_demo SVG plots."""
    path = Path(csv_path)
    output_dir = Path(save_dir) if save_dir is not None else path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows_csv(path)
    save_validity_demo_plot(rows, output_dir, csv_path=path, generator="validity_demo")
    save_validity_demo_budget_curve_plots(output_dir, source_dir=path.parent)
    print(f"Wrote validity_demo plots under: {output_dir}")
    return rows


def save_validity_demo_budget_curve_plots(output_dir: Path, source_dir: Path | None = None) -> None:
    csv_dir = source_dir if source_dir is not None else output_dir
    budget_rows = _read_optional_csv(csv_dir / "benchmark_budget_curves.csv")
    _save_certified_fraction_budget_plot(
        budget_rows,
        output_dir / "validity_demo_budget_curve.svg",
        "validity_demo certified fraction by budget",
        [
            ("Joint row-column shared MILP", "Shared MILP", "validity_full_harmful_sequence_per_prompt", "radius_derived"),
            ("TPA sequence baseline", "TPA max-token sequence", "validity_full_harmful_sequence_per_prompt", "radius_derived"),
            ("DPA weakest harmful-token diagnostic", "DPA weakest harmful token", "weakest_harmful_token_not_full_sequence_validity", "radius_derived"),
        ],
    )
    result_rows = _read_optional_csv(csv_dir / "benchmark_results.csv")
    _save_focused_plot(
        result_rows,
        output_dir / "validity_demo_certificate_vs_K.svg",
        title="validity_demo certificate vs K",
        axis_name="K",
        y_label="Mean certified budget B*",
        metrics={
            "DPA weakest harmful-token diagnostic": "dpa_val_row_weak_q1",
            "TPA sequence baseline": "tpa_val_sequence_q1",
            "Row-only shared MILP": "row_validity",
            "Joint row-column shared MILP": "row_col_val_q1",
        },
    )


def write_validity_demo_audit(path: Path, rows: list[dict[str, object]], csv_path: Path, plotted: list[str], skipped: list[str], generator: str) -> None:
    gap_series = _mean_series_by_axis(rows, "L", "validity_gap_joint_minus_tpa_q1")
    if gap_series is None:
        gap_series = _computed_gap_series(rows, "row_col_val_q1", "tpa_val_sequence_q1")
    tpa_minus_dpa_series = _mean_series_by_axis(rows, "L", "validity_gap_tpa_minus_dpa_weak_q1")
    if tpa_minus_dpa_series is None:
        tpa_minus_dpa_series = _computed_gap_series(rows, "tpa_val_sequence_q1", "dpa_val_row_weak_q1")
    gap_observed = any((_numeric_value(row.get("validity_gap_joint_minus_tpa_q1")) or 0) > 0 for row in rows)
    if not gap_observed:
        gap_observed = any((_safe_numeric_difference(row.get("row_col_val_q1"), row.get("tpa_val_sequence_q1")) or 0) > 0 for row in rows)
    gap_grows = False
    if gap_series is not None and len(gap_series[1]) >= 2:
        gap_grows = gap_series[1][-1] > gap_series[1][0]
    ordering_observed = all(
        (_numeric_value(row.get("dpa_val_row_weak_q1")) or float("inf"))
        < (_numeric_value(row.get("tpa_val_sequence_q1")) or -float("inf"))
        < (_numeric_value(row.get("row_col_val_q1")) or -float("inf"))
        for row in rows
    )
    lines = [
        f"{generator} audit",
        "",
        f"CSV used: {csv_path}",
        f"Generator: {generator}",
        f"K values: {_sorted_unique_values(rows, 'K')}",
        f"N values: {_sorted_unique_values(rows, 'N')}",
        f"L values: {_sorted_unique_values(rows, 'L')}",
        f"T values: {_sorted_unique_values(rows, 'T')}",
        f"Series plotted: {plotted}",
        f"Series skipped: {skipped or ['none']}",
        "",
        "TPA sequence values by L:",
        *_format_mean_series(rows, "tpa_val_sequence_q1"),
        "",
        "DPA weakest harmful-token diagnostic values by L:",
        *_format_mean_series(rows, "dpa_val_row_weak_q1"),
        "",
        "Row-only shared MILP values by L:",
        *_format_mean_series(rows, "row_validity"),
        "",
        "Joint row-column shared MILP values by L:",
        *_format_mean_series(rows, "row_col_val_q1"),
        "",
        "Joint minus TPA gap by L:",
        *(_format_series(gap_series) if gap_series is not None else ["- no numeric rows"]),
        "",
        "TPA minus DPA weakest-token diagnostic gap by L:",
        *(_format_series(tpa_minus_dpa_series) if tpa_minus_dpa_series is not None else ["- no numeric rows"]),
        "",
        f"Expected DPA < TPA < joint ordering observed: {ordering_observed}",
        f"Expected gap observed: {gap_observed}",
        f"Gap grows with L: {gap_grows}",
        "",
        "Explanation:",
        f"{generator} is artificial and controlled. It is not intended to model a natural language distribution.",
        "TPA is count-based and sees each harmful target token as individually cheap from aggregate counts.",
        "The joint row-column shared MILP is shard-aware and must use one shared poisoned-shard allocation across target positions.",
        "The demo assigns cheap target-token attacks to different shard groups, so the full harmful sequence requires more shared poisoned shards than TPA's count-only sequence baseline suggests.",
    ]
    path.write_text("\n".join(lines))


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


def print_certificate_table(results: list[CertificateResult]) -> None:
    """Print certificate names, budgets, and solver statuses."""
    print()
    print(f"{'Certificate':34} B*   Status")
    print("-" * 52)
    for result in results:
        b_star = "NA" if result.B_star is None else str(result.B_star)
        print(f"{result.name:34} {b_star:>2}   {result.status_name}")


def _print_instance_summary(
    data: ToyData, K: int, N: int, L: int, T: int, delta_stab: float, delta_val: float, target_bias: float, seed: int, influence_mode: str
) -> None:
    margins = stability_margins(data.stab_counts, data.clean_pred, data.runner_up)
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


def _print_sweep_table(headers: list[str], rows: list[list[object]]) -> None:
    widths = [max(len(str(header)), *(len(str(row[idx])) for row in rows)) for idx, header in enumerate(headers)]
    header_line = " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers))
    print(header_line)
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)))


def _resolve_objective_flags(
    *,
    objective_family: str,
    make_budget_curves: bool,
    make_horizon_curves: bool,
    make_stability_objectives: bool | None,
    make_validity_objectives: bool | None,
    make_stability_budget_curves: bool | None,
    make_validity_budget_curves: bool | None,
    make_stability_horizon_curves: bool | None,
    make_validity_horizon_curves: bool | None,
) -> dict[str, bool]:
    if objective_family == "full":
        flags = {
            "make_stability_objectives": True,
            "make_validity_objectives": True,
            "make_stability_budget_curves": make_budget_curves,
            "make_validity_budget_curves": make_budget_curves,
            "make_stability_horizon_curves": make_horizon_curves,
            "make_validity_horizon_curves": make_horizon_curves,
        }
    elif objective_family == "validity_only":
        flags = {
            "make_stability_objectives": False,
            "make_validity_objectives": True,
            "make_stability_budget_curves": False,
            "make_validity_budget_curves": make_budget_curves,
            "make_stability_horizon_curves": False,
            "make_validity_horizon_curves": make_horizon_curves,
        }
    else:
        raise ValueError(f"Unknown objective_family: {objective_family}")
    for key, value in {
        "make_stability_objectives": make_stability_objectives,
        "make_validity_objectives": make_validity_objectives,
        "make_stability_budget_curves": make_stability_budget_curves,
        "make_validity_budget_curves": make_validity_budget_curves,
        "make_stability_horizon_curves": make_stability_horizon_curves,
        "make_validity_horizon_curves": make_validity_horizon_curves,
    }.items():
        if value is not None:
            flags[key] = value
    return flags


def _compute_benchmark_baselines(data: ToyData, make_stability_objectives: bool, make_validity_objectives: bool) -> dict[str, int | float]:
    if make_stability_objectives and make_validity_objectives:
        return compute_reference_baselines(data)
    rows: dict[str, int | float] = {}
    if make_validity_objectives:
        validity_cell_budgets = _cell_validity_budgets(data)
        targeted_validity_cell_budgets = _targeted_validity_token_budgets(data)
        phrase_row_budgets = _atomic_phrase_validity_row_budgets(data)
        row_validity_weak_radii = validity_cell_budgets.min(axis=1)
        independent_validity_row_costs = validity_cell_budgets.sum(axis=1)
        rows.update(
            {
                "dpa_val_cell_min": int(np.min(validity_cell_budgets)),
                "dpa_val_row_weak_q1": int(np.min(row_validity_weak_radii)),
                "dpa_val_row_weak_qN": int(np.max(row_validity_weak_radii)),
                "raw_dpa_val_min_cell": int(np.min(validity_cell_budgets)),
                "tpa_val_cell_min": int(np.min(targeted_validity_cell_budgets)),
                **aggregate_tpa_sequence_baselines(targeted_validity_cell_budgets),
                "independent_val_sequence_q1": int(np.min(independent_validity_row_costs)),
                "independent_val_sequence_qN": int(independent_validity_row_costs.sum()),
                "independent_val_q1": int(np.min(independent_validity_row_costs)),
                "independent_val_qN": int(independent_validity_row_costs.sum()),
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
    stability_competitor_mode: str = "all",
    make_stability_objectives: bool = True,
    make_validity_objectives: bool = True,
) -> list[CertificateResult]:
    """Solve the certificate set stored in benchmark CSV rows."""
    N = data.stab_votes.shape[1]
    L = data.stab_votes.shape[2]
    results: list[CertificateResult] = []
    if make_stability_objectives:
        results.extend(
            [
                solve_row_stability(data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, competitor_mode=stability_competitor_mode),
                solve_col_stability(data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, competitor_mode=stability_competitor_mode),
                solve_structured_stability(
                    data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, q_rows=1, r_cols=1, competitor_mode=stability_competitor_mode
                ),
                solve_structured_stability(
                    data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, q_rows=1, r_cols=L, competitor_mode=stability_competitor_mode
                ),
                solve_structured_stability(
                    data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, q_rows=N, r_cols=1, competitor_mode=stability_competitor_mode
                ),
                solve_structured_stability(
                    data.stab_votes, data.stab_counts, data.clean_pred, data.runner_up, data.influence, q_rows=N, r_cols=L, competitor_mode=stability_competitor_mode
                ),
            ]
        )
    if make_validity_objectives:
        results.extend(
            [
                solve_row_validity(data.val_votes, data.val_counts, data.target, T, data.influence),
                solve_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, definition="full_column"),
                solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=1),
                solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=N),
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


def prefix_horizons_from_token_radii(token_radii: np.ndarray, budget: int) -> np.ndarray:
    """Return longest certified prefix per row under the strict ``B < B*`` rule."""
    token_radii = np.asarray(token_radii, dtype=float)
    if token_radii.ndim != 2:
        raise ValueError("token_radii must have shape (N, L)")
    horizons = np.zeros(token_radii.shape[0], dtype=np.int64)
    for i in range(token_radii.shape[0]):
        horizon = 0
        for radius in token_radii[i]:
            if budget < radius:
                horizon += 1
            else:
                break
        horizons[i] = horizon
    return horizons


def targeted_validity_prefix_horizons_from_token_radii(token_radii: np.ndarray, budget: int) -> np.ndarray:
    """Return TPA-style harmful-prefix validity horizons for each row.

    For a harmful prefix to be forced, every target token in that prefix must be
    forced. The prefix is therefore certified impossible whenever at least one
    token in the prefix has radius greater than the attack budget.
    """
    token_radii = np.asarray(token_radii, dtype=float)
    if token_radii.ndim != 2:
        raise ValueError("token_radii must have shape (N, L)")
    horizons = np.zeros(token_radii.shape[0], dtype=np.int64)
    for i in range(token_radii.shape[0]):
        prefix_radii = np.maximum.accumulate(token_radii[i])
        certified_prefixes = np.flatnonzero(budget < prefix_radii)
        if certified_prefixes.size:
            horizons[i] = int(certified_prefixes[-1] + 1)
    return horizons


def compute_radius_derived_budget_curve_rows(
    data: ToyData,
    T: int,
    budgets: Iterable[int],
    metadata: dict[str, object],
    stability_competitor_mode: str = "all",
    make_stability_curves: bool = True,
    make_validity_curves: bool = True,
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
                    _shared_stability_row_radii(data, r_cols=1, competitor_mode=stability_competitor_mode),
                ),
                (
                    "Shared MILP",
                    "stability_full_sequence_per_prompt",
                    _shared_stability_row_radii(data, r_cols=data.stab_votes.shape[2], competitor_mode=stability_competitor_mode),
                ),
            ]
        )
    if make_validity_curves:
        validity_cell_budgets = _cell_validity_budgets(data)
        targeted_validity_cell_budgets = _targeted_validity_token_budgets(data)
        phrase_row_budgets = _atomic_phrase_validity_row_budgets(data)
        curves.extend(
            [
                (
                    "DPA weakest harmful token",
                    "weakest_harmful_token_not_full_sequence_validity",
                    validity_cell_budgets.min(axis=1),
                ),
                (
                    "TPA max-token sequence",
                    "validity_full_harmful_sequence_per_prompt",
                    targeted_validity_cell_budgets.max(axis=1),
                ),
                (
                    "Shared MILP",
                    "validity_full_harmful_sequence_per_prompt",
                    _shared_validity_row_radii(data),
                ),
                (
                    "Atomic phrase aggregation",
                    "validity_full_harmful_sequence_per_prompt",
                    phrase_row_budgets,
                ),
                (
                    "Independent composition",
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


def compute_direct_damage_curve_rows(
    data: ToyData,
    T: int,
    budgets: Iterable[int],
    metadata: dict[str, object],
    stability_competitor_mode: str = "all",
    make_stability_curves: bool = True,
    make_validity_curves: bool = True,
) -> list[dict[str, object]]:
    """Build long-format fixed-budget shared-MILP damage curve rows."""
    objectives = []
    if make_stability_curves:
        objectives.extend(
            [
                (
                    "stability_one_token_per_prompt",
                    lambda budget: maximize_attacked_rows_stability(
                        data.stab_votes,
                        data.stab_counts,
                        data.clean_pred,
                        data.runner_up,
                        data.influence,
                        budget=budget,
                        row_requirement="any_token",
                        competitor_mode=stability_competitor_mode,
                    ),
                ),
                (
                    "stability_full_sequence_per_prompt",
                    lambda budget: maximize_attacked_rows_stability(
                        data.stab_votes,
                        data.stab_counts,
                        data.clean_pred,
                        data.runner_up,
                        data.influence,
                        budget=budget,
                        row_requirement="full_sequence",
                        competitor_mode=stability_competitor_mode,
                    ),
                ),
            ]
        )
    if make_validity_curves:
        objectives.append(
            (
                "validity_full_harmful_sequence_per_prompt",
                lambda budget: maximize_attacked_rows_validity(
                    data.val_votes,
                    data.val_counts,
                    data.target,
                    T,
                    data.influence,
                    budget=budget,
                    row_requirement="full_sequence",
                ),
            )
        )
    rows: list[dict[str, object]] = []
    N = data.stab_votes.shape[1]
    for budget in budgets:
        for objective, solve in objectives:
            result = solve(int(budget))
            rows.append(_damage_curve_row(metadata, result, objective=objective, num_rows=N))
    return rows


def compute_horizon_curve_rows(
    data: ToyData,
    budgets: Iterable[int],
    metadata: dict[str, object],
    make_stability_curves: bool = True,
    make_validity_curves: bool = True,
) -> list[dict[str, object]]:
    """Build long-format prefix horizon summaries for fixed budgets."""
    curves = []
    if make_stability_curves:
        curves.append(
            (
                "DPA stability horizon",
                "stability_clean_prefix",
                _cell_stability_budgets(data),
                prefix_horizons_from_token_radii,
            )
        )
    if make_validity_curves:
        curves.append(
            (
                "TPA validity horizon",
                "validity_harmful_target_prefix",
                _targeted_validity_token_budgets(data),
                targeted_validity_prefix_horizons_from_token_radii,
            )
        )
    rows: list[dict[str, object]] = []
    for method, objective, token_radii, horizon_fn in curves:
        for budget in budgets:
            horizons = horizon_fn(token_radii, int(budget))
            full_horizon = int(token_radii.shape[1])
            full_horizon_fraction = float(np.mean(horizons == full_horizon))
            rows.append(
                {
                    **metadata,
                    "budget": int(budget),
                    "method": method,
                    "objective": objective,
                    "mean_horizon": float(np.mean(horizons)),
                    "median_horizon": float(np.median(horizons)),
                    "min_horizon": int(np.min(horizons)),
                    "max_horizon": int(np.max(horizons)),
                    "mean_horizon_fraction": float(np.mean(horizons) / full_horizon),
                    "median_horizon_fraction": float(np.median(horizons) / full_horizon),
                    "min_horizon_fraction": float(np.min(horizons) / full_horizon),
                    "max_horizon_fraction": float(np.max(horizons) / full_horizon),
                    "max_possible_horizon": full_horizon,
                    "full_horizon_certified_fraction": full_horizon_fraction,
                    "certified_fraction_full_horizon": full_horizon_fraction,
                }
            )
    return rows


def compute_structured_stability_grid(data: ToyData, competitor_mode: str = "all") -> np.ndarray:
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
                data.runner_up,
                data.influence,
                q_rows=q_rows,
                r_cols=r_cols,
                competitor_mode=competitor_mode,
            )
            grid[q_rows - 1, r_cols - 1] = -1 if result.B_star is None else result.B_star
    return grid


def compute_validity_q_curve(data: ToyData, T: int) -> list[int]:
    """Compute validity budgets for one harmful sequence through all prompts."""
    N = data.val_votes.shape[1]
    values = []
    for q_rows in range(1, N + 1):
        result = solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=q_rows)
        values.append(-1 if result.B_star is None else result.B_star)
    return values


def _shared_stability_row_radii(data: ToyData, r_cols: int, competitor_mode: str = "all") -> np.ndarray:
    _, N, _ = data.stab_votes.shape
    radii = np.full(N, np.nan, dtype=float)
    for i in range(N):
        row_slice = slice(i, i + 1)
        result = solve_structured_stability(
            data.stab_votes[:, row_slice, :],
            data.stab_counts[row_slice, :, :],
            data.clean_pred[row_slice, :],
            data.runner_up[row_slice, :],
            data.influence[:, row_slice, :],
            q_rows=1,
            r_cols=r_cols,
            competitor_mode=competitor_mode,
        )
        if result.is_optimal and result.B_star is not None:
            radii[i] = float(result.B_star)
    return radii


def _shared_validity_row_radii(data: ToyData) -> np.ndarray:
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
    stability_competitor_mode: str,
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
        "stability_competitor_mode": stability_competitor_mode,
    }


def _damage_curve_row(metadata: dict[str, object], result: DamageResult, objective: str, num_rows: int) -> dict[str, object]:
    max_attacked_rows = result.max_attacked_rows
    if max_attacked_rows is None and result.objective_value is not None:
        max_attacked_rows = int(round(result.objective_value))
    attacked_fraction = None if max_attacked_rows is None else max_attacked_rows / num_rows
    bound_type = "exact" if result.is_optimal else "feasible_attacked_lower_bound"
    return {
        **metadata,
        "budget": result.budget,
        "method": "Shared MILP",
        "objective": objective,
        "curve_type": "direct_damage_milp",
        "max_attacked_rows": "" if max_attacked_rows is None else max_attacked_rows,
        "max_attacked_cells": result.max_attacked_cells,
        "attacked_fraction": "" if attacked_fraction is None else float(attacked_fraction),
        "certified_fraction": "" if attacked_fraction is None else float(1.0 - attacked_fraction),
        "status_name": result.status_name,
        "is_optimal": result.is_optimal,
        "certified_fraction_is_exact": result.is_optimal,
        "bound_type": bound_type,
        "objective_value": "" if result.objective_value is None else result.objective_value,
        "lower_bound": "" if result.lower_bound is None else result.lower_bound,
        "upper_bound": "" if result.upper_bound is None else result.upper_bound,
        "mip_gap": "" if result.mip_gap is None else result.mip_gap,
        "runtime_sec": "" if result.runtime_sec is None else result.runtime_sec,
    }


def _save_focused_plot(rows: list[dict[str, object]], path: Path, title: str, axis_name: str, y_label: str, metrics: dict[str, str]) -> None:
    """Save a mean-by-axis line plot for selected benchmark metric columns."""
    series = {}
    for label, metric in metrics.items():
        metric_series = _mean_series_by_axis(rows, axis_name, metric)
        if metric_series is None:
            print(f"Warning: skipping {path.name} curve '{label}' because metric '{metric}' is missing or empty.")
            continue
        series[label] = metric_series
    if not series:
        print(f"Warning: skipped {path.name}; none of the requested metric columns were available.")
        return
    _save_line_plot(path, title, _axis_label(axis_name), y_label, series)


def _save_stability_mode_comparison_plots(rows: list[dict[str, object]], output_dir: Path) -> None:
    """Save exact-vs-runner-up stability difference and runtime plots."""
    diff_series = _mean_series_by_axis(rows, "L", "diff")
    if diff_series is not None:
        _save_line_plot_svg(
            output_dir / "stability_mode_diff_by_L.svg",
            "Stability competitor mode difference",
            _axis_label("L"),
            "Mean runner-up minus all-competitor B*",
            {"runner-up minus all-competitor": diff_series},
        )

    grouped: dict[float, list[float]] = {}
    for row in rows:
        x_value = _numeric_value(row.get("L"))
        all_runtime = _numeric_value(row.get("all_runtime_sec"))
        runner_runtime = _numeric_value(row.get("runner_up_runtime_sec"))
        if x_value is None or all_runtime is None or runner_runtime is None or runner_runtime == 0:
            continue
        grouped.setdefault(x_value, []).append(all_runtime / runner_runtime)
    if grouped:
        xs, ys = [], []
        for x in sorted(grouped):
            xs.append(x)
            ys.append(float(np.mean(grouped[x])))
        _save_line_plot_svg(
            output_dir / "stability_mode_runtime_by_L.svg",
            "Stability competitor mode runtime ratio",
            _axis_label("L"),
            "Mean all-competitor / runner-up runtime",
            {"runtime ratio": (xs, ys)},
        )


def _save_certified_fraction_budget_plot(
    rows: list[dict[str, object]],
    path: Path,
    title: str,
    selections: list[tuple[str, str, str, str]],
) -> None:
    if not rows:
        print(f"Warning: skipped {path.name}; no budget curve CSV rows were available.")
        return
    series = {}
    for label, method, objective, curve_type in selections:
        selected = [
            row
            for row in rows
            if row.get("method") == method and row.get("objective") == objective and row.get("curve_type") == curve_type
        ]
        if curve_type == "direct_damage_milp":
            exact_rows = [row for row in selected if row.get("certified_fraction_is_exact") is True or row.get("is_optimal") is True]
            skipped = len(selected) - len(exact_rows)
            if skipped:
                print(f"Warning: skipping {skipped} non-optimal direct-damage row(s) for {path.name} curve '{label}'.")
            selected = exact_rows
        metric_series = _mean_series_by_axis(selected, "budget", "certified_fraction")
        if metric_series is None:
            print(f"Warning: skipping {path.name} curve '{label}' because matching rows are missing.")
            continue
        xs, ys = metric_series
        series[label] = (xs, [100.0 * y for y in ys])
    if not series:
        print(f"Warning: skipped {path.name}; none of the requested curves were available.")
        return
    _save_line_plot(path, title, "Poisoned shard budget", "Certified fraction (%)", series)


def _select_curve_rows(rows: list[dict[str, object]], method: str, objective: str, curve_type: str) -> list[dict[str, object]]:
    return [row for row in rows if row.get("method") == method and row.get("objective") == objective and row.get("curve_type") == curve_type]


def _audit_direct_damage_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    diagnostics: list[dict[str, object]] = []
    if not rows:
        return diagnostics
    print()
    print("Direct damage checks")
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in ["seed", "K", "N", "L", "T", "delta_stab", "delta_val", "target_bias", "influence_mode", "stability_competitor_mode", "objective"])
        grouped.setdefault(key, []).append(row)
        attacked = _numeric_value(row.get("max_attacked_rows"))
        n_rows = _numeric_value(row.get("N"))
        if attacked is not None and n_rows is not None and not 0 <= attacked <= n_rows:
            diagnostic = {"check": "damage_bounds", "objective": row.get("objective"), "budget": row.get("budget"), "max_attacked_rows": attacked, "N": n_rows}
            diagnostics.append(diagnostic)
            print(f"- warning: max_attacked_rows outside [0,N]: {diagnostic}")
    nonoptimal = [row for row in rows if not (row.get("is_optimal") is True or row.get("certified_fraction_is_exact") is True)]
    if nonoptimal:
        print(f"- non-optimal direct damage rows: {len(nonoptimal)}; certified_fraction is a bound, not an exact percentage.")
        diagnostics.append({"check": "nonoptimal_direct_damage", "rows": len(nonoptimal)})

    for key, group_rows in grouped.items():
        sorted_rows = sorted(group_rows, key=lambda row: _numeric_value(row.get("budget")) or -1)
        previous_attacked = None
        previous_certified = None
        for row in sorted_rows:
            attacked = _numeric_value(row.get("attacked_fraction"))
            certified = _numeric_value(row.get("certified_fraction"))
            if attacked is not None and previous_attacked is not None and attacked + 1e-9 < previous_attacked:
                diagnostics.append({"check": "attacked_fraction_monotonicity", "key": key, "budget": row.get("budget")})
            if certified is not None and previous_certified is not None and certified > previous_certified + 1e-9:
                diagnostics.append({"check": "certified_fraction_monotonicity", "key": key, "budget": row.get("budget")})
            previous_attacked = attacked if attacked is not None else previous_attacked
            previous_certified = certified if certified is not None else previous_certified
    monotone_violations = [item for item in diagnostics if str(item.get("check", "")).endswith("monotonicity")]
    print(f"- monotonicity violations: {len(monotone_violations)}")
    return diagnostics


def _audit_horizon_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    diagnostics: list[dict[str, object]] = []
    if not rows:
        return diagnostics
    print()
    print("Horizon checks")
    for row in rows:
        min_horizon = _numeric_value(row.get("min_horizon"))
        max_horizon = _numeric_value(row.get("max_horizon"))
        max_possible = _numeric_value(row.get("max_possible_horizon"))
        full_fraction = _numeric_value(row.get("full_horizon_certified_fraction"))
        if full_fraction is None:
            full_fraction = _numeric_value(row.get("certified_fraction_full_horizon"))
        if (
            min_horizon is not None
            and max_horizon is not None
            and max_possible is not None
            and not (0 <= min_horizon <= max_horizon <= max_possible)
        ):
            diagnostics.append({"check": "horizon_bounds", "method": row.get("method"), "budget": row.get("budget")})
        if full_fraction is not None and not 0 <= full_fraction <= 1:
            diagnostics.append({"check": "horizon_full_fraction_bounds", "method": row.get("method"), "budget": row.get("budget")})
    violations = [item for item in diagnostics if str(item.get("check", "")).startswith("horizon")]
    print(f"- horizon bound violations: {len(violations)}")
    return diagnostics


def _print_stability_mode_comparison_summary(rows: list[dict[str, object]]) -> None:
    comparable = [row for row in rows if _numeric_value(row.get("diff")) is not None]
    optimal = [row for row in comparable if row.get("all_is_optimal") is True and row.get("runner_up_is_optimal") is True]
    diffs = [_numeric_value(row.get("diff")) for row in optimal]
    diffs = [diff for diff in diffs if diff is not None]
    violations = [diff for diff in diffs if diff < 0]
    all_runtimes = [_numeric_value(row.get("all_runtime_sec")) for row in comparable]
    runner_runtimes = [_numeric_value(row.get("runner_up_runtime_sec")) for row in comparable]
    all_runtimes = [value for value in all_runtimes if value is not None]
    runner_runtimes = [value for value in runner_runtimes if value is not None]

    print()
    print("Stability competitor mode comparison summary")
    print(f"Compared rows: {len(comparable)}")
    print(f"Optimal pairs: {len(optimal)}")
    if diffs:
        print(f"Fraction diff == 0: {sum(diff == 0 for diff in diffs) / len(diffs):.3f}")
        print(f"Mean diff: {float(np.mean(diffs)):.3f}")
        print(f"Max diff: {float(np.max(diffs)):.3f}")
    else:
        print("Fraction diff == 0: NA")
        print("Mean diff: NA")
        print("Max diff: NA")
    print(f"Violations runner_up < all: {len(violations)}")
    print(f"Mean all-competitor runtime: {float(np.mean(all_runtimes)):.6f}s" if all_runtimes else "Mean all-competitor runtime: NA")
    print(f"Mean runner-up runtime: {float(np.mean(runner_runtimes)):.6f}s" if runner_runtimes else "Mean runner-up runtime: NA")


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
    if axis_name == "target_bias":
        return "target bias"
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


def _save_heatmap_svg(
    matrix: np.ndarray,
    path: Path,
    title: str,
    fmt: str = ".0f",
    x_label: str = "token column j",
    y_label: str = "prompt row i",
    colorbar_label: str | None = None,
) -> None:
    """Save a lightweight SVG heatmap without external plotting dependencies."""
    rows, cols = matrix.shape
    cell = 54
    left = 70
    right = 96 if colorbar_label else 40
    width = left + cols * cell + right
    title_lines = _wrap_svg_text(title, max_chars=max(36, int((width - 56) / 7.2)))
    title_font_size = 13
    title_line_height = 17
    title_start_y = 26
    top = title_start_y + title_line_height * len(title_lines) + 22
    height = top + rows * cell + 56
    values = matrix.astype(float)
    vmin = float(np.min(values))
    vmax = float(np.max(values))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left + cols * cell / 2:.1f}" y="{height - 12}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">{_xml_escape(x_label)}</text>',
        f'<text x="16" y="{top + rows * cell / 2:.1f}" transform="rotate(-90 16 {top + rows * cell / 2:.1f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">{_xml_escape(y_label)}</text>',
    ]
    for idx, line in enumerate(title_lines):
        y = title_start_y + idx * title_line_height
        svg.insert(
            2 + idx,
            f'<text x="{width / 2:.1f}" y="{y}" text-anchor="middle" font-family="Arial, sans-serif" font-size="{title_font_size}" fill="#111827">{_xml_escape(line)}</text>',
        )
    for j in range(cols):
        svg.append(f'<text x="{left + j * cell + cell / 2}" y="{top - 10}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">{j + 1 if colorbar_label else j}</text>')
    for i in range(rows):
        svg.append(f'<text x="{left - 12}" y="{top + i * cell + cell / 2 + 4}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#374151">{i + 1 if colorbar_label else i}</text>')
        for j in range(cols):
            value = float(values[i, j])
            color = _heat_color(value, vmin, vmax)
            x = left + j * cell
            y = top + i * cell
            svg.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
            svg.append(
                f'<text x="{x + cell / 2}" y="{y + cell / 2 + 5}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#ffffff">{format(value, fmt)}</text>'
            )
    if colorbar_label is not None:
        bar_x = left + cols * cell + 28
        bar_y = top
        bar_h = rows * cell
        for step in range(24):
            t0 = step / 24
            y = bar_y + bar_h * (1 - (step + 1) / 24)
            value = vmin + (vmax - vmin) * t0
            svg.append(f'<rect x="{bar_x}" y="{y:.1f}" width="14" height="{bar_h / 24 + 0.8:.1f}" fill="{_heat_color(value, vmin, vmax)}"/>')
        svg.append(f'<text x="{bar_x + 20}" y="{bar_y + 4}" font-family="Arial, sans-serif" font-size="10" fill="#374151">{vmax:.0f}</text>')
        svg.append(f'<text x="{bar_x + 20}" y="{bar_y + bar_h:.1f}" font-family="Arial, sans-serif" font-size="10" fill="#374151">{vmin:.0f}</text>')
        label_x = bar_x + 58
        label_y = bar_y + bar_h / 2
        svg.append(
            f'<text x="{label_x}" y="{label_y:.1f}" transform="rotate(-90 {label_x} {label_y:.1f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">{_xml_escape(colorbar_label)}</text>'
        )
    svg.append("</svg>")
    path.write_text("\n".join(svg))


def _save_line_plot(path: Path, title: str, x_label: str, y_label: str, series: dict[str, tuple[list[float], list[float]]]) -> None:
    """Save a thesis-facing line plot.

    PNG is used for the cleaned report outputs. If matplotlib is unavailable,
    the lightweight SVG renderer is used as a compatibility fallback.
    """
    if path.suffix.lower() != ".png":
        _save_line_plot_svg(path, title, x_label, y_label, series)
        return
    try:
        cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "provably_secure_nlg_plot_cache"
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local plotting deps.
        fallback = path.with_suffix(".svg")
        print(f"Warning: matplotlib unavailable ({exc}); writing SVG fallback {fallback.name}.")
        _save_line_plot_svg(fallback, title, x_label, y_label, series)
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=220)
    for label, (xs, ys) in series.items():
        color = CANONICAL_COLORS.get(label)
        linestyle = "--" if "baseline" in label else "-"
        marker = "s" if "PHD" in label else "o"
        ax.plot(xs, ys, label=label, color=color, linestyle=linestyle, marker=marker, linewidth=2.2, markersize=4.8)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_line_plot_svg(path: Path, title: str, x_label: str, y_label: str, series: dict[str, tuple[list[float], list[float]]]) -> None:
    """Save a lightweight multi-series SVG line plot."""
    width = 1160
    title_lines = _wrap_svg_text(title, max_chars=max(48, int((width - 80) / 7.5)))
    title_font_size = 14
    title_line_height = 18
    title_start_y = 28
    top = title_start_y + title_line_height * len(title_lines) + 24
    height = max(540 + title_line_height * max(0, len(title_lines) - 1), 430 + 22 * len(series) + title_line_height * max(0, len(title_lines) - 1))
    left, right, bottom = 72, 360, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_x = [x for xs, _ in series.values() for x in xs]
    all_y = [y for _, ys in series.values() for y in ys]
    if not all_x or not all_y:
        return
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(0.0, min(all_y)), max(all_y)
    if xmin == xmax:
        xmin -= 1
        xmax += 1
    if ymin == ymax:
        ymax += 1
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2", "#be123c", "#4d7c0f", "#9333ea", "#475569"]

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (y - ymin) / (ymax - ymin) * plot_h

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#374151">{_xml_escape(x_label)}</text>',
        f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#374151">{_xml_escape(y_label)}</text>',
    ]
    for idx, line in enumerate(title_lines):
        y = title_start_y + idx * title_line_height
        svg.insert(
            2 + idx,
            f'<text x="{width / 2}" y="{y}" text-anchor="middle" font-family="Arial, sans-serif" font-size="{title_font_size}" fill="#111827">{_xml_escape(line)}</text>',
        )
    for tick in range(5):
        y_value = ymin + (ymax - ymin) * tick / 4
        y = sy(y_value)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#374151">{y_value:.1f}</text>')
    for tick in range(5):
        x_value = xmin + (xmax - xmin) * tick / 4
        x = sx(x_value)
        svg.append(f'<text x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">{x_value:.2g}</text>')

    for idx, (name, (xs, ys)) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in zip(xs, ys):
            svg.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>')
        legend_y = top + 18 + idx * 20
        svg.append(f'<rect x="{left + plot_w + 24}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        svg.append(f'<text x="{left + plot_w + 42}" y="{legend_y}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{_xml_escape(name)}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg))


def _wrap_svg_text(text: str, max_chars: int) -> list[str]:
    """Wrap SVG title text into short lines without requiring SVG text layout."""
    parts = text.split(" | ", maxsplit=1)
    if len(parts) == 2:
        prefix, suffix = parts
        chunks = [prefix]
        words = suffix.split()
    else:
        chunks = []
        words = text.split()

    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = word
    if current:
        chunks.append(current)
    return chunks or [text]


def _heat_color(value: float, vmin: float, vmax: float) -> str:
    if vmax == vmin:
        t = 0.5
    else:
        t = (value - vmin) / (vmax - vmin)
    start = np.array([37, 99, 235], dtype=float)
    mid = np.array([20, 184, 166], dtype=float)
    end = np.array([234, 179, 8], dtype=float)
    if t < 0.5:
        rgb = start + (mid - start) * (t / 0.5)
    else:
        rgb = mid + (end - mid) * ((t - 0.5) / 0.5)
    return "#" + "".join(f"{int(round(c)):02x}" for c in rgb)


def _xml_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _parse_float_list(value: str) -> list[float]:
    if isinstance(value, list):
        return value
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_int_list(value: str) -> list[int]:
    if isinstance(value, list):
        return value
    return [int(part.strip()) for part in value.split(",") if part.strip()]


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
    for line_number, raw_line in enumerate(config_path.read_text().splitlines(), start=1):
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ConfigError(f"invalid config line {line_number} in {config_path}: expected `key: value`")
        key, value = line.split(":", maxsplit=1)
        key = key.strip()
        if not key:
            raise ConfigError(f"invalid config line {line_number} in {config_path}: empty key")
        if key == "preset":
            raise ConfigError(f"field `preset` in {config_path} is not supported in YAML configs; use `name` for metadata")
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
    generator = _require_str(config, config_path, "generator", {"toy", "validity_demo"})
    benchmark_config: dict[str, object] = {
        "Ks": _require_number_list(config, config_path, "K_values", int),
        "Ns": _require_number_list(config, config_path, "N_values", int),
        "lengths": _require_number_list(config, config_path, "L_values", int),
        "Ts": _require_number_list(config, config_path, "T_values", int),
        "delta_stabs": _require_number_list(config, config_path, "delta_stab_values", float),
        "delta_vals": _require_number_list(config, config_path, "delta_val_values", float),
        "target_biases": _require_number_list(config, config_path, "target_bias_values", float),
        "generator": generator,
        "seed": _require_int(config, config_path, "seed"),
        "budget_max": _require_int(config, config_path, "budget_max"),
        "save_dir": _require_str(config, config_path, "output_dir"),
        "influence_mode": _require_str(config, config_path, "influence_mode", {"dense", "row-local", "column-local", "validity_demo"}),
        "stability_competitor_mode": _require_str(config, config_path, "stability_competitor_mode", {"all", "runner_up"}),
        "objective_family": _require_str(config, config_path, "objective_family", {"full", "validity_only"}),
        "make_budget_curves": _require_bool(config, config_path, "make_budget_curves"),
        "make_damage_curves": _require_bool(config, config_path, "make_damage_curves"),
        "make_horizon_curves": _require_bool(config, config_path, "make_horizon_curves"),
    }
    for key in [
        "make_stability_objectives",
        "make_validity_objectives",
        "make_stability_budget_curves",
        "make_validity_budget_curves",
        "make_stability_horizon_curves",
        "make_validity_horizon_curves",
    ]:
        if key in config:
            value = config[key]
            if not isinstance(value, bool):
                raise ConfigError(f"field `{key}` in {config_path} must be a boolean")
            benchmark_config[key] = value
    if generator == "validity_demo":
        benchmark_config.update(
            {
                "group_size": _require_int(config, config_path, "group_size"),
                "target_gap": _require_int(config, config_path, "target_gap"),
                "overlap": _require_int(config, config_path, "overlap"),
            }
        )
    return benchmark_config


def _estimate_solve_counts(
    Ks: Iterable[int],
    Ns: Iterable[int],
    Ls: Iterable[int],
    Ts: Iterable[int],
    delta_stabs: Iterable[float],
    delta_vals: Iterable[float],
    target_biases: Iterable[float],
    budget_max: int,
    objective_flags: dict[str, bool],
) -> dict[str, int]:
    counts = {
        "instances": 0,
        "stability_objective_solves": 0,
        "validity_objective_solves": 0,
        "per_row_stability_solves": 0,
        "per_row_validity_solves": 0,
        "direct_damage_stability_solves": 0,
        "direct_damage_validity_solves": 0,
    }
    for K in Ks:
        for N in Ns:
            for _L in Ls:
                for _T in Ts:
                    for _delta_stab in delta_stabs:
                        for _delta_val in delta_vals:
                            for _target_bias in target_biases:
                                counts["instances"] += 1
                                budget_count = min(int(K), int(budget_max)) + 1
                                if objective_flags["make_stability_objectives"]:
                                    counts["stability_objective_solves"] += 6
                                if objective_flags["make_validity_objectives"]:
                                    counts["validity_objective_solves"] += 4
                                if objective_flags["make_stability_budget_curves"]:
                                    counts["per_row_stability_solves"] += 2 * int(N)
                                if objective_flags["make_validity_budget_curves"]:
                                    counts["per_row_validity_solves"] += int(N)
                                if objective_flags["make_stability_objectives"]:
                                    counts["direct_damage_stability_solves"] += 2 * budget_count
                                if objective_flags["make_validity_objectives"]:
                                    counts["direct_damage_validity_solves"] += budget_count
    if not objective_flags.get("make_damage_curves", False):
        counts["direct_damage_stability_solves"] = 0
        counts["direct_damage_validity_solves"] = 0
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
) -> None:
    Ks, Ns, Ls, Ts = list(Ks), list(Ns), list(Ls), list(Ts)
    delta_stabs, delta_vals, target_biases = list(delta_stabs), list(delta_vals), list(target_biases)
    counts = _estimate_solve_counts(Ks, Ns, Ls, Ts, delta_stabs, delta_vals, target_biases, budget_max, objective_flags)
    total_stability = counts["stability_objective_solves"] + counts["per_row_stability_solves"] + counts["direct_damage_stability_solves"]
    total_validity = counts["validity_objective_solves"] + counts["per_row_validity_solves"] + counts["direct_damage_validity_solves"]
    print("Benchmark dry run")
    print(f"generator: {generator}")
    print(f"output_dir: {save_dir}")
    print(f"objective_family: {objective_family}")
    print(f"stability objectives: {'enabled' if objective_flags['make_stability_objectives'] else 'disabled'}")
    print(f"validity objectives: {'enabled' if objective_flags['make_validity_objectives'] else 'disabled'}")
    print(f"stability budget curves: {'enabled' if objective_flags['make_stability_budget_curves'] else 'disabled'}")
    print(f"validity budget curves: {'enabled' if objective_flags['make_validity_budget_curves'] else 'disabled'}")
    print(f"stability horizon curves: {'enabled' if objective_flags['make_stability_horizon_curves'] else 'disabled'}")
    print(f"validity horizon curves: {'enabled' if objective_flags['make_validity_horizon_curves'] else 'disabled'}")
    print(f"damage curves: {'enabled' if objective_flags['make_damage_curves'] else 'disabled'}")
    print(f"expanded benchmark instances: {counts['instances']}")
    if verbose:
        print(f"K values: {Ks}")
        print(f"N values: {Ns}")
        print(f"L values: {Ls}")
        print(f"T values: {Ts}")
        print(f"delta_stab values: {delta_stabs}")
        print(f"delta_val values: {delta_vals}")
        print(f"target_bias values: {target_biases}")
    print("estimated Gurobi solves:")
    print(f"  estimated stability solves: {total_stability}")
    print(f"  estimated validity solves: {total_validity}")


BENCHMARK_PRESETS: dict[str, dict[str, list[float] | list[int]]] = {
    "smoke": {
        "Ks": [8],
        "Ns": [2],
        "lengths": [6],
        "Ts": [4],
        "delta_stabs": [0.2],
        "delta_vals": [0.2],
        "target_biases": [0.3],
    },
    "small": {
        "Ks": [4, 6, 8],
        "Ns": [2, 3],
        "lengths": [2, 3, 4],
        "Ts": [4, 6],
        "delta_stabs": [0.2],
        "delta_vals": [0.2],
        "target_biases": [0.3],
    },
    "medium": {
        "Ks": [4, 6, 8, 10],
        "Ns": [2, 3, 4],
        "lengths": [2, 3, 4, 5],
        "Ts": [4, 6, 8],
        "delta_stabs": [0.0, 0.2],
        "delta_vals": [0.0, 0.2],
        "target_biases": [0.1, 0.3],
    },
    "large": {
        "Ks": [5, 10, 15, 20, 25],
        "Ns": [3, 5, 7, 9, 11],
        "lengths": [3, 6, 9, 12],
        "Ts": [3, 6, 9, 12],
        "delta_stabs": [0.0, 0.25, 0.5],
        "delta_vals": [0.0, 0.25, 0.5],
        "target_biases": [0.2],
    },
}


def _benchmark_preset(name: str) -> dict[str, list[float] | list[int]]:
    try:
        return BENCHMARK_PRESETS[name]
    except KeyError as exc:  # pragma: no cover - argparse guards this in normal use.
        raise ValueError(f"Unknown benchmark preset: {name}") from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for toy experiment workflows."""
    parser = argparse.ArgumentParser(description="Toy row/column certificate experiments.")
    parser.add_argument(
        "command",
        choices=[
            "sanity",
            "visualize",
            "benchmark",
            "plot-csv",
            "plot-validity-demo",
            "audit-curves",
            "sweep-delta",
            "sweep-length",
            "sweep-prompts",
            "compare-stability-modes",
        ],
    )
    parser.add_argument("--K", type=int, default=7)
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument("--delta-stab", type=float, default=None)
    parser.add_argument("--delta-val", type=float, default=None)
    parser.add_argument("--delta-stabs", type=_parse_float_list, default=None)
    parser.add_argument("--delta-vals", type=_parse_float_list, default=None)
    parser.add_argument("--target-bias", type=float, default=None)
    parser.add_argument("--target-biases", type=_parse_float_list, default=None)
    parser.add_argument("--generator", choices=["toy", "validity_demo"], default="toy")
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--target-gap", type=int, default=1)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--influence-mode", choices=["dense", "row-local", "column-local", "validity_demo"], default="dense")
    parser.add_argument("--stability-competitor-mode", choices=["all", "runner_up"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preset", choices=sorted(BENCHMARK_PRESETS), default=None, help="Benchmark preset used when explicit grid ranges are omitted.")
    parser.add_argument("--deltas", type=_parse_float_list, default=None)
    parser.add_argument("--Ks", type=_parse_int_list, default=None)
    parser.add_argument("--Ns", type=_parse_int_list, default=None)
    parser.add_argument("--lengths", type=_parse_int_list, default=None)
    parser.add_argument("--prompts", type=_parse_int_list, default=None)
    parser.add_argument("--Ts", type=_parse_int_list, default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--csv", default="outputs/results/benchmark_results.csv")
    parser.add_argument("--csv-dir", default="outputs/results")
    parser.add_argument("--show-grid", action="store_true")
    parser.add_argument("--make-plots", action="store_true", help="Also render benchmark plots after running Gurobi.")
    parser.add_argument("--budget-max", type=int, default=15, help="Maximum poisoned-shard budget for fixed-budget curve CSVs.")
    parser.add_argument("--make-budget-curves", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make-damage-curves", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make-horizon-curves", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--objective-family", choices=["full", "validity_only"], default="full")
    parser.add_argument("--make-stability-objectives", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--make-validity-objectives", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--make-stability-budget-curves", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--make-validity-budget-curves", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--make-stability-horizon-curves", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--make-validity-horizon-curves", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--config", default=None, help="YAML benchmark config. Preferred for benchmark runs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print solve estimates without running Gurobi.")
    parser.add_argument("--verbose", action="store_true", help="Print resolved dry-run grid values.")
    return parser


def _resolve_benchmark_grid(args: argparse.Namespace) -> dict[str, list[float] | list[int]]:
    preset_name = args.preset or "medium"
    preset = _benchmark_preset(preset_name)
    target_bias_values = args.target_biases if args.target_biases is not None else preset["target_biases"]
    if args.target_bias is not None:
        target_bias_values = [args.target_bias]
    return {
        "Ks": args.Ks if args.Ks is not None else preset["Ks"],
        "Ns": args.Ns if args.Ns is not None else preset["Ns"],
        "lengths": args.lengths if args.lengths is not None else preset["lengths"],
        "Ts": args.Ts if args.Ts is not None else preset["Ts"],
        "delta_stabs": args.delta_stabs if args.delta_stabs is not None else args.deltas if args.deltas is not None else preset["delta_stabs"],
        "delta_vals": args.delta_vals if args.delta_vals is not None else args.deltas if args.deltas is not None else preset["delta_vals"],
        "target_biases": target_bias_values,
    }


def main() -> None:
    """Entry point for ``python -m toy_certificate.experiments``."""
    args = build_parser().parse_args()
    delta_stab = args.delta if args.delta_stab is None else args.delta_stab
    delta_val = args.delta if args.delta_val is None else args.delta_val
    target_bias = args.target_bias if args.target_bias is not None else 0.2
    if args.command == "sanity":
        run_sanity(
            K=args.K,
            N=args.N,
            L=args.L,
            T=args.T,
            delta_stab=delta_stab,
            delta_val=delta_val,
            target_bias=target_bias,
            seed=args.seed,
            influence_mode=args.influence_mode,
            stability_competitor_mode=args.stability_competitor_mode,
            show_grid=args.show_grid,
            save_dir=(args.save_dir or "outputs/smoke") if args.show_grid else None,
        )
    elif args.command == "visualize":
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
            save_dir=args.save_dir or "outputs/smoke",
            stability_competitor_mode=args.stability_competitor_mode,
        )
    elif args.command == "benchmark":
        if args.config is not None:
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
                stability_competitor_mode=config["stability_competitor_mode"],
                seed=config["seed"],
                save_dir=config["save_dir"],
                make_plots=args.make_plots,
                budget_max=config["budget_max"],
                make_budget_curves=config["make_budget_curves"],
                make_damage_curves=config["make_damage_curves"],
                make_horizon_curves=config["make_horizon_curves"],
                objective_family=config["objective_family"],
                make_stability_objectives=config.get("make_stability_objectives"),
                make_validity_objectives=config.get("make_validity_objectives"),
                make_stability_budget_curves=config.get("make_stability_budget_curves"),
                make_validity_budget_curves=config.get("make_validity_budget_curves"),
                make_stability_horizon_curves=config.get("make_stability_horizon_curves"),
                make_validity_horizon_curves=config.get("make_validity_horizon_curves"),
                dry_run=args.dry_run,
                verbose=args.verbose,
                generator=config["generator"],
                group_size=config.get("group_size", args.group_size),
                target_gap=config.get("target_gap", args.target_gap),
                overlap=config.get("overlap", args.overlap),
            )
            return
        print("WARNING: direct CLI benchmark parameters are deprecated; use --config path/to/config.yaml instead.")
        grid = _resolve_benchmark_grid(args)
        benchmark_scale(
            Ks=grid["Ks"],
            Ns=grid["Ns"],
            Ls=grid["lengths"],
            Ts=grid["Ts"],
            delta_stabs=grid["delta_stabs"],
            delta_vals=grid["delta_vals"],
            target_biases=grid["target_biases"],
            influence_mode=args.influence_mode,
            stability_competitor_mode=args.stability_competitor_mode,
            seed=args.seed,
            save_dir=args.save_dir or "outputs/results",
            make_plots=args.make_plots,
            budget_max=args.budget_max,
            make_budget_curves=args.make_budget_curves,
            make_damage_curves=args.make_damage_curves,
            make_horizon_curves=args.make_horizon_curves,
            objective_family=args.objective_family,
            make_stability_objectives=args.make_stability_objectives,
            make_validity_objectives=args.make_validity_objectives,
            make_stability_budget_curves=args.make_stability_budget_curves,
            make_validity_budget_curves=args.make_validity_budget_curves,
            make_stability_horizon_curves=args.make_stability_horizon_curves,
            make_validity_horizon_curves=args.make_validity_horizon_curves,
            dry_run=args.dry_run,
            verbose=args.verbose,
            generator=args.generator,
            group_size=args.group_size,
            target_gap=args.target_gap,
            overlap=args.overlap,
        )
    elif args.command == "plot-csv":
        plot_benchmark_csv(args.csv, save_dir=args.save_dir)
    elif args.command == "plot-validity-demo":
        plot_validity_demo_csv(args.csv, save_dir=args.save_dir)
    elif args.command == "audit-curves":
        audit_curve_csvs(args.csv_dir)
    elif args.command == "sweep-delta":
        sweep_delta(
            K=args.K,
            N=args.N,
            L=args.L,
            T=args.T,
            deltas=args.deltas if args.deltas is not None else [0.0, 0.2, 0.4],
            seed=args.seed,
            stability_competitor_mode=args.stability_competitor_mode,
        )
    elif args.command == "sweep-length":
        sweep_length(
            K=args.K,
            N=args.N,
            lengths=args.lengths if args.lengths is not None else [2, 4],
            T=args.T,
            delta=args.delta,
            seed=args.seed,
            stability_competitor_mode=args.stability_competitor_mode,
        )
    elif args.command == "sweep-prompts":
        sweep_prompts(
            K=args.K,
            prompts=args.prompts if args.prompts is not None else [1, 2, 4, 8],
            L=args.L,
            T=args.T,
            delta=args.delta,
            seed=args.seed,
            stability_competitor_mode=args.stability_competitor_mode,
        )
    elif args.command == "compare-stability-modes":
        compare_stability_modes(
            Ks=args.Ks if args.Ks is not None else _benchmark_preset("small")["Ks"],
            Ns=args.Ns if args.Ns is not None else _benchmark_preset("small")["Ns"],
            Ls=args.lengths if args.lengths is not None else _benchmark_preset("small")["lengths"],
            Ts=args.Ts if args.Ts is not None else _benchmark_preset("small")["Ts"],
            deltas=args.deltas if args.deltas is not None else [0.0, 0.2, 0.4],
            target_bias=args.target_bias if args.target_bias is not None else 0.2,
            influence_mode=args.influence_mode,
            seed=args.seed,
            save_dir=args.save_dir or "outputs/results/stability_mode_comparison",
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
