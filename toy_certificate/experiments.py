"""Experiment CLI, benchmark generation, baselines, and SVG plotting.

This module is the orchestration layer for the first-party toy certificate
implementation. It builds synthetic :class:`toy_certificate.data.ToyData`
instances, calls the shared MILP solvers, writes benchmark CSVs, computes
DPA/phrase-DPA/independent-composition baselines, and renders report-facing SVG
plots. The external ``phd_reference/`` tree is not imported or modified here.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np

from .data import ToyData, generate_toy_votes, stability_margins
from .milp import (
    CertificateResult,
    solve_col_stability,
    solve_col_validity,
    solve_row_col_stability,
    solve_row_col_validity,
    solve_row_stability,
    solve_row_validity,
    solve_structured_stability,
)


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
    deltas: Iterable[float],
    target_bias: float,
    influence_mode: str,
    stability_competitor_mode: str,
    seed: int,
    save_dir: str,
    make_plots: bool = False,
) -> list[dict[str, object]]:
    """Generate benchmark CSV rows for the configured parameter grid.

    CSV generation is intentionally separate from plotting. Stability solver
    rows record ``stability_competitor_mode`` so exact all-competitor runs can be
    distinguished from runner-up approximation runs.
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
                        start = perf_counter()
                        results = _solve_benchmark_certificates(data, T, stability_competitor_mode=stability_competitor_mode)
                        runtime_total = perf_counter() - start
                        row = {
                            "K": K,
                            "N": N,
                            "L": L,
                            "T": T,
                            "delta_stab": delta,
                            "delta_val": delta,
                            "target_bias": target_bias,
                            "seed": seed,
                            "influence_mode": influence_mode,
                            "stability_competitor_mode": stability_competitor_mode,
                            "runtime_gurobi_total": f"{runtime_total:.6f}",
                        }
                        for result in results:
                            metric_name = _csv_metric_name(result.name, N, L)
                            row[metric_name] = result.B_star
                        _fill_degenerate_corner_columns(row)
                        row.update(compute_reference_baselines(data))
                        rows.append(row)
                        print(
                            "bench "
                            f"K={K} N={N} L={L} T={T} delta={delta}: "
                            + ", ".join(f"{result.name}={result.B_star}" for result in results)
                        )

    csv_path = output_dir / "benchmark_results.csv"
    _write_rows_csv(csv_path, rows)
    print()
    print(f"Wrote benchmark CSV: {csv_path}")
    if make_plots:
        save_benchmark_plots(rows, output_dir)
        print(f"Wrote benchmark plots under: {output_dir}")
    else:
        print(f"Skipped plots. Replot later with: python -m toy_certificate.experiments plot-csv --csv {csv_path}")
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


def _fill_degenerate_corner_columns(row: dict[str, object]) -> None:
    if int(row["L"]) == 1:
        if "row_col_stab_q1_r1" in row:
            row.setdefault("row_col_stab_q1_rL", row["row_col_stab_q1_r1"])
        if "row_col_stab_qN_r1" in row:
            row.setdefault("row_col_stab_qN_rL", row["row_col_stab_qN_r1"])
    if int(row["N"]) == 1:
        if "row_col_stab_q1_r1" in row:
            row.setdefault("row_col_stab_qN_r1", row["row_col_stab_q1_r1"])
        if "row_col_stab_q1_rL" in row:
            row.setdefault("row_col_stab_qN_rL", row["row_col_stab_q1_rL"])
        if "row_col_val_q1" in row:
            row.setdefault("row_col_val_qN", row["row_col_val_q1"])


def plot_benchmark_csv(csv_path: str, save_dir: str | None = None) -> list[dict[str, object]]:
    """Read a benchmark CSV and regenerate report-facing SVG plots."""
    path = Path(csv_path)
    output_dir = Path(save_dir) if save_dir is not None else path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows_csv(path)
    save_benchmark_plots(rows, output_dir)
    print(f"Wrote benchmark plots under: {output_dir}")
    return rows


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


def save_benchmark_plots(rows: list[dict[str, object]], output_dir: Path) -> None:
    """Write report-facing benchmark plots plus compatibility aggregate plots."""
    if not rows:
        return

    report_plots = [
        (
            output_dir / "validity_one_prompt_by_L.svg",
            "Validity: forcing one harmful sequence",
            "L",
            "Mean poison budget B*",
            {
                "DPA weakest harmful token": "dpa_val_row_weak_q1",
                "phrase-DPA full sequence": "phrase_dpa_val_q1",
                "shared MILP full sequence": "row_col_val_q1",
                "independent full sequence": "independent_val_sequence_q1",
            },
        ),
        (
            output_dir / "validity_all_prompts_by_L.svg",
            "Validity: forcing harmful sequences for all prompts",
            "L",
            "Mean poison budget B*",
            {
                "shared MILP: all harmful sequences": "row_col_val_qN",
                "independent: all harmful sequences": "independent_val_sequence_qN",
                "DPA weakest harmful token per prompt": "dpa_val_row_weak_qN",
            },
        ),
        (
            output_dir / "stability_one_prompt_by_L.svg",
            "Stability: changing one prompt",
            "L",
            "Mean poison budget B*",
            {
                "DPA weakest token": "dpa_stab_row_radius_q1",
                "shared MILP: one token": "row_col_stab_q1_r1",
                "shared MILP: full sequence": "row_col_stab_q1_rL",
            },
        ),
        (
            output_dir / "stability_all_prompts_by_L.svg",
            "Stability: changing all prompts",
            "L",
            "Mean poison budget B*",
            {
                "shared MILP: one token per prompt": "row_col_stab_qN_r1",
                "shared MILP: full matrix": "row_col_stab_qN_rL",
                "DPA weakest token per prompt": "dpa_stab_row_radius_qN",
            },
        ),
    ]
    for path, title, axis_name, y_label, metrics in report_plots:
        _save_focused_plot(rows, path, title=title, axis_name=axis_name, y_label=y_label, metrics=metrics)

    _save_independent_validity_overestimate_plot(rows, output_dir / "validity_independent_overestimate_by_L.svg")
    _save_independent_stability_overestimate_plot(rows, output_dir / "stability_independent_overestimate_by_L.svg")

    # Backwards-compatible aggregate plots. The objective-specific plots above are
    # preferred for report figures.
    _save_focused_plot(
        rows,
        output_dir / "validity_scaling_by_L.svg",
        title="Validity scaling with sequence length",
        axis_name="L",
        y_label="Mean poison budget B*",
        metrics={
            "DPA weakest harmful token": "dpa_val_row_weak_q1",
            "phrase-DPA full sequence": "phrase_dpa_val_q1",
            "shared MILP full sequence": "row_col_val_q1",
            "shared MILP: all harmful sequences": "row_col_val_qN",
        },
    )
    _save_focused_plot(
        rows,
        output_dir / "stability_structured_by_L.svg",
        title="Structured stability scaling with sequence length",
        axis_name="L",
        y_label="Mean poison budget B*",
        metrics={
            "DPA weakest token": "dpa_stab_row_radius_q1",
            "DPA weakest token per prompt": "dpa_stab_row_radius_qN",
            "one prompt, one token": "row_col_stab_q1_r1",
            "one prompt, full sequence": "row_col_stab_q1_rL",
            "all prompts, one token each": "row_col_stab_qN_r1",
            "all prompts, full matrix": "row_col_stab_qN_rL",
        },
    )
    _save_focused_plot(
        rows,
        output_dir / "validity_bias_sweep.svg",
        title="Validity sensitivity to harmful-prefix target bias",
        axis_name="target_bias",
        y_label="Mean poison budget B*",
        metrics={
            "DPA weakest harmful token": "dpa_val_row_weak_q1",
            "DPA weakest harmful token per prompt": "dpa_val_row_weak_qN",
            "shared MILP: one harmful sequence": "row_col_val_q1",
            "shared MILP: harmful sequences for all prompts": "row_col_val_qN",
            "phrase-DPA full sequence": "phrase_dpa_val_q1",
        },
    )
    check_monotonicity_diagnostics(rows, output_dir)


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


def _solve_benchmark_certificates(data: ToyData, T: int, stability_competitor_mode: str = "all") -> list[CertificateResult]:
    """Solve the certificate set stored in benchmark CSV rows."""
    N = data.stab_votes.shape[1]
    L = data.stab_votes.shape[2]
    return [
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
        solve_row_validity(data.val_votes, data.val_counts, data.target, T, data.influence),
        solve_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, definition="full_column"),
        solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=1),
        solve_row_col_validity(data.val_votes, data.val_counts, data.target, T, data.influence, q_rows=N),
    ]


def compute_reference_baselines(data: ToyData) -> dict[str, int]:
    """Compute DPA, phrase-DPA, and independent-composition baseline columns."""
    stability_cell_budgets = _cell_stability_budgets(data)
    validity_cell_budgets = _cell_validity_budgets(data)
    phrase_row_budgets = _phrase_dpa_validity_row_budgets(data)
    row_stability_radii = stability_cell_budgets.min(axis=1)
    row_validity_weak_radii = validity_cell_budgets.min(axis=1)
    independent_stability_row_costs = stability_cell_budgets.sum(axis=1)
    independent_validity_row_costs = validity_cell_budgets.sum(axis=1)
    return {
        "raw_dpa_stab_min_cell": int(np.min(_phd_margin_stability_budgets(data))),
        "dpa_stab_cell_min": int(np.min(stability_cell_budgets)),
        "dpa_stab_row_radius_q1": int(np.min(row_stability_radii)),
        "dpa_stab_row_radius_qN": int(np.max(row_stability_radii)),
        "dpa_val_cell_min": int(np.min(validity_cell_budgets)),
        "dpa_val_row_weak_q1": int(np.min(row_validity_weak_radii)),
        "dpa_val_row_weak_qN": int(np.max(row_validity_weak_radii)),
        "raw_dpa_val_min_cell": int(np.min(validity_cell_budgets)),
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


def _cell_stability_budgets(data: ToyData) -> np.ndarray:
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
                competitor_budgets.append(_min_budget_from_contributions(deficit, contributions))
            budgets[i, j] = min(competitor_budgets)
    return budgets


def _cell_validity_budgets(data: ToyData) -> np.ndarray:
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
            budgets[i, j] = _min_budget_satisfying_all(deficits, contribs, ignored_class=h)
    return budgets


def _phd_margin_stability_budgets(data: ToyData) -> np.ndarray:
    margins = stability_margins(data.stab_counts, data.clean_pred, data.runner_up)
    return ((margins + 1) // 2).astype(np.int64)


def _phrase_dpa_validity_row_budgets(data: ToyData) -> np.ndarray:
    """Compute phrase-DPA budgets treating each generated sequence as one label."""
    K, N, L = data.val_votes.shape
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
        budgets[i] = _min_budget_satisfying_all(deficits, contribs)
    return budgets


def _min_budget_from_contributions(deficit: int, contributions: list[int]) -> int:
    if deficit <= 0:
        return 0
    running = 0
    for budget, contribution in enumerate(sorted(contributions, reverse=True), start=1):
        running += contribution
        if running >= deficit:
            return budget
    return len(contributions) + 1


def _min_budget_satisfying_all(deficits: np.ndarray, contribs: np.ndarray, ignored_class: int | None = None) -> int:
    active_deficits = deficits.copy()
    if ignored_class is not None:
        active_deficits[ignored_class] = 0
    if np.all(active_deficits <= 0):
        return 0

    useful = np.flatnonzero(np.any(contribs > 0, axis=1))
    if useful.size == 0:
        return int(contribs.shape[0] + 1)

    for budget in range(1, useful.size + 1):
        for subset in combinations(useful.tolist(), budget):
            if np.all(contribs[list(subset)].sum(axis=0) >= active_deficits):
                return budget
    return int(contribs.shape[0] + 1)


def _write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write heterogeneous benchmark dictionaries while preserving first-seen column order."""
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    _save_line_plot_svg(path, title, _axis_label(axis_name), y_label, series)


def _save_independent_stability_overestimate_plot(rows: list[dict[str, object]], path: Path) -> None:
    """Plot how much independent-composition stability overestimates shared MILP."""
    numerator_metric = _first_available_metric(rows, ["independent_stab_full_row_qN", "independent_stab_qN_rL"])
    denominator_metric = "row_col_stab_qN_rL"
    if numerator_metric is None:
        print(f"Warning: skipped {path.name}; no independent full-matrix stability metric was available.")
        return
    xs, ys, y_label = _ratio_or_difference_series(rows, "L", numerator_metric, denominator_metric)
    if not xs:
        print(f"Warning: skipped {path.name}; metrics '{numerator_metric}' and '{denominator_metric}' had no paired numeric values.")
        return

    _save_line_plot_svg(
        path,
        "Independent composition overestimate",
        _axis_label("L"),
        y_label,
        {"independent overestimate factor": (xs, ys)},
    )


def _save_independent_validity_overestimate_plot(rows: list[dict[str, object]], path: Path) -> None:
    """Plot how much independent-composition validity overestimates shared MILP."""
    numerator_metric = "independent_val_sequence_q1"
    denominator_metric = "row_col_val_q1"
    xs, ys, y_label = _ratio_or_difference_series(rows, "L", numerator_metric, denominator_metric)
    if not xs:
        print(f"Warning: skipped {path.name}; metrics '{numerator_metric}' and '{denominator_metric}' had no paired numeric values.")
        return

    _save_line_plot_svg(
        path,
        "Independent composition overestimate",
        _axis_label("L"),
        y_label,
        {"independent overestimate factor": (xs, ys)},
    )


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


def _ratio_or_difference_series(
    rows: list[dict[str, object]], axis_name: str, numerator_metric: str, denominator_metric: str
) -> tuple[list[float], list[float], str]:
    """Build a ratio series, falling back to differences when denominators are zero."""
    paired: list[tuple[float, float, float]] = []
    has_zero_denominator = False
    for row in rows:
        x_value = _numeric_value(row.get(axis_name))
        numerator = _numeric_value(row.get(numerator_metric))
        denominator = _numeric_value(row.get(denominator_metric))
        if x_value is None or numerator is None or denominator is None:
            continue
        has_zero_denominator = has_zero_denominator or denominator == 0
        paired.append((x_value, numerator, denominator))

    grouped: dict[float, list[float]] = {}
    for x_value, numerator, denominator in paired:
        if has_zero_denominator:
            value = numerator - denominator
        else:
            value = numerator / denominator
        grouped.setdefault(x_value, []).append(value)

    xs, ys = [], []
    for x in sorted(grouped):
        xs.append(x)
        ys.append(float(np.mean(grouped[x])))
    y_label = "Mean independent overestimate factor" if not has_zero_denominator else "Mean independent minus shared MILP B*"
    return xs, ys, y_label


def _first_available_metric(rows: list[dict[str, object]], metric_names: list[str]) -> str | None:
    for metric_name in metric_names:
        if any(_numeric_value(row.get(metric_name)) is not None for row in rows):
            return metric_name
    return None


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


def check_monotonicity_diagnostics(rows: list[dict[str, object]], output_dir: Path | None = None) -> list[dict[str, object]]:
    """Check expected monotonicity relations in benchmark CSV rows."""
    checks = [
        ("stability", "row_col_stab_q1_r1", "<=", "row_col_stab_q1_rL"),
        ("stability", "row_col_stab_q1_r1", "<=", "row_col_stab_qN_r1"),
        ("stability", "row_col_stab_q1_rL", "<=", "row_col_stab_qN_rL"),
        ("stability", "row_col_stab_qN_r1", "<=", "row_col_stab_qN_rL"),
        ("validity", "row_col_val_q1", "<=", "row_col_val_qN"),
    ]
    violations: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        for objective, left_metric, relation, right_metric in checks:
            left = _numeric_value(row.get(left_metric))
            right = _numeric_value(row.get(right_metric))
            if left is None or right is None:
                continue
            if left <= right:
                continue
            violation = {
                "row_index": idx,
                "objective": objective,
                "left_metric": left_metric,
                "relation": relation,
                "right_metric": right_metric,
                "left_value": left,
                "right_value": right,
            }
            for key in ["K", "N", "L", "T", "delta_stab", "delta_val", "target_bias", "seed", "influence_mode"]:
                if key in row:
                    violation[key] = row[key]
            violations.append(violation)

    print(f"Monotonicity diagnostics: {len(violations)} violation(s).")
    if output_dir is not None and violations:
        path = output_dir / "monotonicity_violations.csv"
        _write_rows_csv(path, violations)
        print(f"Wrote monotonicity violations: {path}")
    return violations


def _read_rows_csv(path: Path) -> list[dict[str, object]]:
    """Read benchmark rows and normalize legacy column aliases."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key in {"K", "N", "L", "T", "seed"}:
                    parsed[key] = int(value)
                elif key == "delta" or _looks_numeric(value):
                    parsed[key] = float(value) if "." in value else int(value)
                else:
                    parsed[key] = value
            _copy_legacy_csv_columns(parsed)
            rows.append(parsed)
    return rows


def _copy_legacy_csv_columns(row: dict[str, object]) -> None:
    legacy_map = {
        "row_col_stability_any_cell": "row_col_stab_q1_r1",
        "row_col_stability_full_row": "row_col_stab_q1_rL",
        "row_col_validity_q1": "row_col_val_q1",
        "row_col_validity_qN": "row_col_val_qN",
        "naive_dpa_stability_full_row": "independent_stab_full_row_q1",
        "naive_dpa_validity_q1": "independent_val_q1",
        "naive_dpa_validity_qN": "independent_val_qN",
        "phd_ref_stability_any_cell": "raw_dpa_stab_min_cell",
        "phd_ref_validity_any_cell": "raw_dpa_val_min_cell",
        "independent_stab_qN_rL": "independent_stab_full_row_qN",
        "independent_val_q1": "independent_val_sequence_q1",
        "independent_val_qN": "independent_val_sequence_qN",
    }
    for old_key, new_key in legacy_map.items():
        if old_key in row and new_key not in row:
            row[new_key] = row[old_key]
    if "raw_dpa_stab_min_cell" in row:
        row.setdefault("dpa_stab_cell_min", row["raw_dpa_stab_min_cell"])
        row.setdefault("dpa_stab_row_radius_q1", row["raw_dpa_stab_min_cell"])
    if "raw_dpa_val_min_cell" in row:
        row.setdefault("dpa_val_cell_min", row["raw_dpa_val_min_cell"])
        row.setdefault("dpa_val_row_weak_q1", row["raw_dpa_val_min_cell"])


def _looks_numeric(value: object) -> bool:
    if value is None:
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True


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
    top = 58
    right = 96 if colorbar_label else 24
    width = left + cols * cell + right
    height = top + rows * cell + 56
    values = matrix.astype(float)
    vmin = float(np.min(values))
    vmax = float(np.max(values))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111827">{_xml_escape(title)}</text>',
        f'<text x="{left + cols * cell / 2:.1f}" y="{height - 12}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">{_xml_escape(x_label)}</text>',
        f'<text x="16" y="{top + rows * cell / 2:.1f}" transform="rotate(-90 16 {top + rows * cell / 2:.1f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">{_xml_escape(y_label)}</text>',
    ]
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


def _save_line_plot_svg(path: Path, title: str, x_label: str, y_label: str, series: dict[str, tuple[list[float], list[float]]]) -> None:
    """Save a lightweight multi-series SVG line plot."""
    width = 1160
    height = max(540, 430 + 22 * len(series))
    left, right, top, bottom = 72, 360, 52, 70
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
        f'<text x="{width / 2}" y="26" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="#111827">{_xml_escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#374151">{_xml_escape(x_label)}</text>',
        f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#374151">{_xml_escape(y_label)}</text>',
    ]
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


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for toy experiment workflows."""
    parser = argparse.ArgumentParser(description="Toy row/column certificate experiments.")
    parser.add_argument(
        "command", choices=["sanity", "visualize", "benchmark", "plot-csv", "sweep-delta", "sweep-length", "sweep-prompts", "compare-stability-modes"]
    )
    parser.add_argument("--K", type=int, default=7)
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument("--delta-stab", type=float, default=None)
    parser.add_argument("--delta-val", type=float, default=None)
    parser.add_argument("--target-bias", type=float, default=0.2)
    parser.add_argument("--influence-mode", choices=["dense", "row-local", "column-local"], default="dense")
    parser.add_argument("--stability-competitor-mode", choices=["all", "runner_up"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deltas", type=_parse_float_list, default=[0.0, 0.2, 0.4])
    parser.add_argument("--Ks", type=_parse_int_list, default=[5, 7, 9])
    parser.add_argument("--Ns", type=_parse_int_list, default=[2, 3, 5])
    parser.add_argument("--lengths", type=_parse_int_list, default=[2, 4])
    parser.add_argument("--prompts", type=_parse_int_list, default=[1, 2, 4, 8])
    parser.add_argument("--Ts", type=_parse_int_list, default=[3, 5])
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--csv", default="toy_results/benchmark_large/benchmark_results.csv")
    parser.add_argument("--show-grid", action="store_true")
    parser.add_argument("--make-plots", action="store_true", help="Also render benchmark plots after running Gurobi.")
    return parser


def main() -> None:
    """Entry point for ``python -m toy_certificate.experiments``."""
    args = build_parser().parse_args()
    delta_stab = args.delta if args.delta_stab is None else args.delta_stab
    delta_val = args.delta if args.delta_val is None else args.delta_val
    if args.command == "sanity":
        run_sanity(
            K=args.K,
            N=args.N,
            L=args.L,
            T=args.T,
            delta_stab=delta_stab,
            delta_val=delta_val,
            target_bias=args.target_bias,
            seed=args.seed,
            influence_mode=args.influence_mode,
            stability_competitor_mode=args.stability_competitor_mode,
            show_grid=args.show_grid,
            save_dir=(args.save_dir or "toy_results") if args.show_grid else None,
        )
    elif args.command == "visualize":
        visualize_instance(
            K=args.K,
            N=args.N,
            L=args.L,
            T=args.T,
            delta_stab=delta_stab,
            delta_val=delta_val,
            target_bias=args.target_bias,
            seed=args.seed,
            influence_mode=args.influence_mode,
            save_dir=args.save_dir or "toy_results",
            stability_competitor_mode=args.stability_competitor_mode,
        )
    elif args.command == "benchmark":
        benchmark_scale(
            Ks=args.Ks,
            Ns=args.Ns,
            Ls=args.lengths,
            Ts=args.Ts,
            deltas=args.deltas,
            target_bias=args.target_bias,
            influence_mode=args.influence_mode,
            stability_competitor_mode=args.stability_competitor_mode,
            seed=args.seed,
            save_dir=args.save_dir or "toy_results",
            make_plots=args.make_plots,
        )
    elif args.command == "plot-csv":
        plot_benchmark_csv(args.csv, save_dir=args.save_dir)
    elif args.command == "sweep-delta":
        sweep_delta(K=args.K, N=args.N, L=args.L, T=args.T, deltas=args.deltas, seed=args.seed, stability_competitor_mode=args.stability_competitor_mode)
    elif args.command == "sweep-length":
        sweep_length(K=args.K, N=args.N, lengths=args.lengths, T=args.T, delta=args.delta, seed=args.seed, stability_competitor_mode=args.stability_competitor_mode)
    elif args.command == "sweep-prompts":
        sweep_prompts(K=args.K, prompts=args.prompts, L=args.L, T=args.T, delta=args.delta, seed=args.seed, stability_competitor_mode=args.stability_competitor_mode)
    elif args.command == "compare-stability-modes":
        compare_stability_modes(
            Ks=args.Ks,
            Ns=args.Ns,
            Ls=args.lengths,
            Ts=args.Ts,
            deltas=args.deltas,
            target_bias=args.target_bias,
            influence_mode=args.influence_mode,
            seed=args.seed,
            save_dir=args.save_dir or "toy_results/stability_mode_comparison",
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
